import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt

from processes.video_writer_process import VideoWriterProcess
from configuration.config import Config, VisualizationStrategy
from perception.helpers import get_roi_bbox_for_video, extract_pipeline_names, stack_images_v3, \
    draw_rois_and_wait
from perception.objects.pipe_data import PipeData
from perception.objects.save_info import SaveInfo
from perception.objects.timingvisualizer import TimingVisualizer
from perception.objects.video_info import VideoInfo, VideoRois
from perception.visualize_data import visualize_data
from processes.multiprocessing_manager import MultiProcessingManager
from ripc import SharedMemoryCircularQueue


def main():
    program_start_time = time.perf_counter()
    mp.set_start_method('spawn')

    timing_visualizer = TimingVisualizer()
    ot = 'Overall Timer'
    se = 'Setup'
    timing_visualizer.start(ot)
    timing_visualizer.start(se, parent=ot)

    print("[Main] Config:", Config.as_json())
    # create a new folder in the recordings directory
    os.makedirs(Config.recordings_dir, exist_ok=True)

    # Extract the name without the extension
    video_name = os.path.splitext(Config.video_name)[0]

    recording_dir_path = os.path.join(Config.recordings_dir, f"{video_name}-{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}")

    os.makedirs(recording_dir_path)

    with open(os.path.join(recording_dir_path, 'config.json'), 'w') as file:
        file.write(Config.as_json())

    keep_running = mp.Value('b', True)
    start_video = mp.Value('b', False)

    mp_manager = MultiProcessingManager(keep_running=keep_running, program_start_time=program_start_time, start_video=start_video)
    mp_manager.start()
    mp_manager.wait_for_setup()

    visualization_queue: SharedMemoryCircularQueue = SharedMemoryCircularQueue.open(Config.visualization_memory_name)

    save_queue = None
    video_writer_process = None
    if Config.save_processed_video:
        save_queue: SharedMemoryCircularQueue = SharedMemoryCircularQueue.create(Config.save_final_memory_name, Config.frame_size, Config.save_queue_element_count)

        save_info = SaveInfo(
            video_path=os.path.join(recording_dir_path,
                                    f"Final_{video_name}.mp4"),
            width=Config.width,
            height=Config.height,
            fps=Config.camera_fps
        )

        video_writer_process = VideoWriterProcess(save_info=save_info, shared_memory_name=Config.save_final_memory_name,
                                                  keep_running=keep_running, program_start_time=program_start_time)
        video_writer_process.start()

    video_rois: VideoRois = get_roi_bbox_for_video(Config.video_name, Config.width, Config.height, Config.roi_config_path)
    video_info = VideoInfo(video_name=Config.video_name, height=Config.height,
                           width=Config.width, video_rois=video_rois)
    pipeline_names = extract_pipeline_names()

    pipe_data_bytes = None
    pipe_data = None
    iteration_counter = 0
    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)
    timing_visualizer.stop(se)

    while True:
        if Config.visualizer_strategy == VisualizationStrategy.NEWEST_FRAME:
            if len(visualization_queue) == 0:
                pipe_data_bytes = None
            else:
                list_pipe_data_bytes = visualization_queue.read_all()
                pipe_data_bytes = list_pipe_data_bytes[-1]
        elif Config.visualizer_strategy == VisualizationStrategy.ALL_FRAMES:
            pipe_data_bytes = visualization_queue.try_read()

        if pipe_data_bytes is not None:
            iteration_counter += 1
            pipe_data: PipeData = pickle.loads(pipe_data_bytes)

            dl = f"Data Lifecycle {pipe_data.last_filter_process_name[0]}"
            tf2 = f"Transfer Data (MM -> Viz) {pipe_data.last_filter_process_name[0]}"

            pipe_data.timing_info.stop(tf2)
            pipe_data.timing_info.stop(dl)
            timing_visualizer.timing_info.append_hierarchy(pipe_data.timing_info, parent_label_of_other=ot)

            # if iteration_counter % Config.fps == 0:
            #     print("Plotting pie charts")
            #     timer.plot_pie_charts(save_path=os.path.join(Config.recordings_dir, 'timings'))

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

            if Config.save_processed_video and save_queue is not None:
                save_queue.try_write(drawn_frame.tobytes())

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('x'):
            if pipe_data is not None:
                draw_rois_and_wait(pipe_data.frame, video_rois)
                cv2.waitKey(0)
            else:
                print("No frame to draw ROIs on")

    timing_visualizer.start("Cleanup", parent=ot)
    cv2.destroyAllWindows()

    keep_running.value = False

    print("[Main] Joining MultiProcessingManager")
    mp_manager.join()
    print("[Main] MultiProcessingManager joined")

    if save_queue is not None and video_writer_process is not None:
        print("[Main] Joining VideoWriterProcess")
        video_writer_process.join()
    print("[Main] Joined VideoWriterProcess")

    timing_visualizer.stop("Cleanup")
    timing_visualizer.stop(ot)
    print(timing_visualizer.timing_info)
    timing_visualizer.plot_pie_charts(save_path=os.path.join(recording_dir_path, 'timings'))
    plt.show()  # Keep the pie chart open

if __name__ == '__main__':
    main()