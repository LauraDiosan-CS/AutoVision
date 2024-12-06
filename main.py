import argparse
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

    timer = TimingVisualizer()
    timer.start('Overall Timer')
    timer.start('Setup', parent='Overall Timer')

    keep_running = mp.Value('b', True)

    mp_manager = MultiProcessingManager(keep_running=keep_running, program_start_time=program_start_time)
    mp_manager.start()
    mp_manager.wait_for_setup()

    visualization_queue: SharedMemoryCircularQueue = SharedMemoryCircularQueue.open(Config.visualization_memory_name)

    save_queue = None
    video_writer_process = None
    if Config.save_processed_video:
        save_queue: SharedMemoryCircularQueue = SharedMemoryCircularQueue.create(Config.save_draw_memory_name, Config.frame_size, Config.save_queue_element_count)

        # Extract the name without the extension
        video_name = os.path.splitext(Config.video_name)[0]

        save_info = SaveInfo(
            video_path=os.path.join(Config.recordings_dir,
                                    f"Processed_{video_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"),
            width=Config.width,
            height=Config.height,
            fps=Config.fps
        )

        video_writer_process = VideoWriterProcess(save_info=save_info, shared_memory_name=Config.save_draw_memory_name,
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
    timer.stop('Setup')

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
            timer.start('Display Frame', parent='Overall Timer')
            pipe_data: PipeData = pickle.loads(pipe_data_bytes)
            pipe_data.timing_info.stop(f"Transfer Data (MM -> Viz) {pipe_data.last_touched_process}")
            pipe_data.timing_info.stop(f"Data Lifecycle {pipe_data.last_touched_process}")
            timer.timing_info.append_hierarchy(pipe_data.timing_info, label="Overall Timer")

            # if iteration_counter % Config.fps == 0:
            #     print("Plotting pie charts")
            #     timer.plot_pie_charts(save_path=os.path.join(Config.recordings_dir, 'timings'))

            frame = visualize_data(video_info=video_info, data=pipe_data)
            if pipe_data.processed_frames is not None and len(pipe_data.processed_frames) > 0:
                squashed_frames = [[frame]]

                for pipeline_name in pipeline_names:
                    if pipeline_name in pipe_data.processed_frames:
                        squashed_frames.append(pipe_data.processed_frames[pipeline_name])
                    else:
                        # add black frame
                        squashed_frames.append([np.zeros((Config.height, Config.width, 3), dtype=np.uint8)])
                final_img = stack_images_v3(1, squashed_frames)
            else:
                final_img = frame

            if Config.save_processed_video and save_queue is not None:
                save_queue.try_write(frame.tobytes())
            cv2.imshow('CarVision', final_img)

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('x'):
            if pipe_data is not None:
                draw_rois_and_wait(pipe_data.frame, video_rois)
                cv2.waitKey(0)
            else:
                print("No frame to draw ROIs on")
        elif key & 0xFF == ord('s'):
            save_dir_path = os.path.join(os.getcwd(), Config.screenshot_dir)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            screenshot_name = f'{Config.video_name}_{timestamp}.jpg'
            screenshot_path = os.path.join(save_dir_path, screenshot_name)
            cv2.imwrite(screenshot_path, pipe_data.frame)

        timer.stop('Display Frame')

    cv2.destroyAllWindows()

    keep_running.value = False
    print("Initiating Termination of MultiProcessingManager")

    if save_queue is not None and video_writer_process is not None:
        print("Joining processed frame saving process")
        video_writer_process.join()
    print("Processed frame saving process joined")

    print("Joining MultiProcessingManager")
    mp_manager.terminate()
    print("MultiProcessingManager joined")

    timer.stop('Overall Timer')
    timer.plot_pie_charts(save_path=os.path.join(Config.recordings_dir, 'timings'))
    plt.show()  # Keep the pie chart open


def update_config(config, args):
    for arg in vars(args):
        if hasattr(config, arg):
            setattr(config, arg, getattr(args, arg))
    Config.print_config()


if __name__ == '__main__':
    args = argparse.ArgumentParser().parse_args()
    update_config(Config, args)
    main()