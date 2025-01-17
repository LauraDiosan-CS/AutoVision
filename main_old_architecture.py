import multiprocessing as mp
import os
import pickle
import time
from copy import deepcopy
from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt

from configuration.config import Config
from perception.filters.base_filter import BaseFilter
from perception.helpers import get_roi_bbox_for_video, extract_pipeline_names, stack_images_v3, \
    draw_rois_and_wait, initialize_config
from perception.objects.pipe_data import PipeData
from perception.objects.timingvisualizer import TimingVisualizer
from perception.objects.video_info import VideoInfo, VideoRois
from perception.visualize_data import visualize_data


class SimpleSequentialFilterProcess(mp.Process):
    __slots__ = ['filters', 'pipe', 'filters', 'artificial_delay']

    def __init__(self, filters: list[BaseFilter], keep_running: mp.Value, pipe: mp.Pipe, artificial_delay: float = 0.0, name: str = None):
        super().__init__(name=name)
        self.filters = filters
        self.keep_running = keep_running
        self.artificial_delay = artificial_delay
        self.pipe = pipe

    def run(self):
        try:
            while self.keep_running.value:
                pipe_data = self.pipe.recv()
                if pipe_data is None:
                    continue

                pickled_data = pickle.dumps(pipe_data)
                print(f"PD size: {len(pickled_data)}")

                pipe_data.last_filter_process_name = self.name

                dl = f"Data Lifecycle {self.name[0]}"
                sd = f"Send Data {self.name[0]}"
                pd = f"Process Data {self.name[0]}"
                rd = f"Receive Data {self.name[0]}"

                pipe_data.timing_info.stop(sd)
                pipe_data.timing_info.start(pd, parent=dl)

                if self.artificial_delay > 0:
                    time.sleep(self.artificial_delay)

                for filter in self.filters:
                    pipe_data: PipeData = filter.process(pipe_data)

                pipe_data.timing_info.stop(pd)
                pipe_data.timing_info.start(rd, parent=dl)

                self.pipe.send(pipe_data)

                del pipe_data
        except Exception as e:
            print(f"[{self.name}] Error: {e}")


def main():
    mp.set_start_method('spawn')

    timing_visualizer = TimingVisualizer()
    ot = 'Overall Timer'
    se = 'Setup'
    timing_visualizer.start(ot)
    timing_visualizer.start(se, parent=ot)

    # print("[Main] Config:", Config.as_json())
    # create a new folder in the recordings directory
    os.makedirs(Config.recordings_dir, exist_ok=True)

    # Extract the name without the extension
    video_name = os.path.splitext(Config.video_name)[0]

    recording_dir_path = os.path.join(Config.recordings_dir, f"{video_name}-{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}")

    os.makedirs(recording_dir_path)

    with open(os.path.join(recording_dir_path, 'config.json'), 'w') as file:
        file.write(Config.as_json())

    video_path = str(os.path.join(Config.videos_dir, Config.video_name))
    capture = cv2.VideoCapture(video_path)

    # Get the actual video dimensions
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine if resizing is necessary
    resize_needed = actual_width != Config.width or actual_height != Config.height

    if resize_needed:
        print(f"Actual video width: {actual_width} height: {actual_height} so resize is needed")

    pipelines, _, _ = initialize_config()

    keep_running = mp.Value('b', True)

    parallel_processes = []

    for (index, pipeline) in enumerate(pipelines):
        match pipeline.name:
            case "lane_detection":
                artificial_delay = 0.0
            case "sign_detection":
                artificial_delay = 0.1
            case "traffic_light_detection":
                artificial_delay = 0.15
            case "pedestrian_detection":
                artificial_delay = 0.2
        artificial_delay = 0.0

        pipe, subprocess_pipe = mp.Pipe()

        process = SimpleSequentialFilterProcess(filters=pipeline.filters,
                                                keep_running=keep_running,
                                                pipe=subprocess_pipe,
                                                artificial_delay=artificial_delay,
                                                name=pipeline.name)
        process.start()
        parallel_processes.append((process, pipe))

    video_rois: VideoRois = get_roi_bbox_for_video(Config.video_name, Config.width, Config.height, Config.roi_config_path)
    video_info = VideoInfo(video_name=Config.video_name, height=Config.height,
                           width=Config.width, video_rois=video_rois)
    pipeline_names = extract_pipeline_names()

    iteration_counter = 0
    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)
    timing_visualizer.stop(se)

    it = 'Iteration'
    pf = 'Process Frame'

    while keep_running.value and capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            print("[Main] Video ended")
            break

        if resize_needed:
            frame = cv2.resize(frame, (Config.width, Config.height), interpolation=cv2.INTER_LINEAR)

        timing_visualizer.start(it, parent=ot)
        iteration_counter+=1

        pipe_data: PipeData = PipeData(frame=frame,
                                  frame_version=iteration_counter,
                                  depth_frame=None,
                                  raw_frame=frame,
                                  creation_time=time.perf_counter(),
                                  last_filter_process_name = "None")

        pipe_data.timing_info.start(pf)

        for process, pipe in parallel_processes:
            pipe_data_copy = deepcopy(pipe_data)
            dl = f"Data Lifecycle {process.name[0]}"
            sd = f"Send Data {process.name[0]}"
            pipe_data_copy.timing_info.start(dl, parent=pf)
            pipe_data_copy.timing_info.start(sd, parent=dl)
            pipe.send(pipe_data_copy)
            del pipe_data_copy

        for process, pipe in parallel_processes:
            dl = f"Data Lifecycle {process.name[0]}"
            rd = f"Receive Data {process.name[0]}"
            new_data = pipe.recv()
            new_data.timing_info.stop(rd)
            new_data.timing_info.stop(dl)
            # print(f"New data timing info: {new_data.timing_info}")
            pipe_data = pipe_data.merge(new_data)
            # print(f"Pipe data timing info: {pipe_data.timing_info}")

        # print(timing_visualizer.timing_info)
        # print(pipe_data.timing_info)
        timing_visualizer.timing_info.append_hierarchy(pipe_data.timing_info, parent_label_of_other=ot)

        drawn_frame = visualize_data(video_info=video_info, data=pipe_data, raw_frame=pipe_data.raw_frame)
        if pipe_data.processed_frames is not None and len(pipe_data.processed_frames) > 0:
            squashed_frames = [[drawn_frame]]

            for pipeline_name in pipeline_names:
                if pipeline_name in pipe_data.processed_frames:
                    squashed_frames.append(pipe_data.processed_frames[pipeline_name])
                else:
                    # add black frame
                    squashed_frames.append([np.zeros((Config.height, Config.width, 3), dtype=np.uint8)])
            stacked_frame = stack_images_v3(1, squashed_frames)
            cv2.imshow('CarVision', stacked_frame)
        else:
            cv2.imshow('CarVision', drawn_frame)

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('x'):
            if pipe_data is not None:
                draw_rois_and_wait(pipe_data.frame, video_rois)
                cv2.waitKey(0)
            else:
                print("No frame to draw ROIs on")

        timing_visualizer.stop(it)

    # timing_visualizer.start("Cleanup", parent=ot)
    cv2.destroyAllWindows()

    keep_running.value = False

    print("Joining all parallel processes")
    for process, pipe in parallel_processes:
        pipe.send(None)
        process.join()
        pipe.close()
    print("All parallel processes joined")

    # timing_visualizer.stop("Cleanup")
    timing_visualizer.stop(ot)
    print(timing_visualizer.timing_info)
    timing_visualizer.plot_pie_charts(save_path=os.path.join(recording_dir_path, 'timings'))
    plt.show()  # Keep the pie chart open



if __name__ == '__main__':
    main()