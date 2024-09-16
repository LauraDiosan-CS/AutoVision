import argparse
import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime

import cv2
from matplotlib import pyplot as plt

from config import Config
from filters.draw_filter import DrawFilter
from helpers.helpers import stack_images_v2, draw_rois_and_wait, get_roi_bbox_for_video, save_frames
from helpers.timingvisualizer import TimingVisualizer
from multiprocessing_manager import MultiProcessingManager
from objects.pipe_data import PipeData
from objects.types.save_info import SaveInfo
from objects.types.video_info import VideoRois, VideoInfo
from ripc import SharedMemoryReader
from videoreaderprocess import VideoReaderProcess


def main():
    timer = TimingVisualizer()
    timer.start('Overall Timer')
    timer.start('Setup', parent='Overall Timer')
    mp.set_start_method('spawn')

    keep_running = mp.Value('b', True)

    video_reader_process = VideoReaderProcess(keep_running=keep_running)
    video_reader_process.start()
    # ripc.V4lSharedMemoryWriter('path', Config.width, "video_name")

    mp_manager = MultiProcessingManager(keep_running)
    mp_manager.start()

    visualizer_memory = SharedMemoryReader(name=Config.composite_pipe_memory_name)

    save_queue = None
    save_process = None
    if Config.save_processed_video:
        save_queue = mp.Queue()

        # Extract the name without the extension
        video_name = os.path.splitext(Config.video_name)[0]

        save_info = SaveInfo(
            video_path=os.path.join(Config.recordings_dir,
                                    f"Processed_{video_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"),
            width=Config.width,
            height=Config.height,
            fps=Config.fps
        )
        save_process = mp.Process(target=save_frames,
                                  args=(save_queue,
                                        save_info))
        save_process.start()

    video_rois: VideoRois = get_roi_bbox_for_video(Config.video_name, Config.roi_config_path)
    video_info = VideoInfo(video_name=Config.video_name, height=Config.height,
                           width=Config.width, video_rois=video_rois)

    draw_filter = DrawFilter(video_info=video_info)

    replay_speed = 1
    pipe_data_bytes = None
    pipe_data = None
    frames_to_skip = 0
    iteration_counter = 0
    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)
    timer.stop('Setup')

    while True:
        for i in range(frames_to_skip + 1): # this doesnt check if there are enough frames to skip
          pipe_data_bytes = visualizer_memory.read()

        if pipe_data_bytes is not None:
            iteration_counter += 1
            timer.start('Display Frame', parent='Overall Timer')
            pipe_data: PipeData = pickle.loads(pipe_data_bytes)
            pipe_data.timings.stop("Transfer Data (MM -> Viz)")
            # print(f"PipeData timings: {pipe_data.timings}")

            timer.start('Process Frame', parent='Overall Timer',
                        extra_time_seconds=time.time() - pipe_data.creation_time)

            timer.start(f"{pipe_data.last_touched_process}", parent="Process Frame")
            timer.start(f"Process Data {pipe_data.last_touched_process}", parent=f"{pipe_data.last_touched_process}",
                        extra_time_seconds=pipe_data.pipeline_execution_time)
            timer.stop(f"Process Data {pipe_data.last_touched_process}")

            timer.start(f"Transfer time {pipe_data.last_touched_process}", parent=f"{pipe_data.last_touched_process}",
                        extra_time_seconds=pipe_data.arrive_time-pipe_data.send_start_time)
            timer.stop(f"Transfer time {pipe_data.last_touched_process}")

            timer.stop(f"{pipe_data.last_touched_process}")
            pipe_data.timings.stop("Data Lifecycle")

            timer.stop('Process Frame')


            if iteration_counter % Config.fps == 0:
                timer.plot_pie_charts(save_path=os.path.join(Config.recordings_dir, 'timings'))
            # end_time = time.time() - pipe_data.creation_time
            # print(f"PipeData with {pipe_data.last_touched_process} took {end_time} seconds equivalent to fps: {1/end_time}")

            if Config.apply_visualizer:
                pipe_data = draw_filter.process(pipe_data)

            if pipe_data.processed_frames is not None:
                squashed_frames = sum(pipe_data.processed_frames.values(), [])
                final_img = stack_images_v2(1, squashed_frames)
            else:
                final_img = pipe_data.frame

            cv2.imshow('CarVision', final_img)

            if Config.save_processed_video and save_queue is not None:
                print("Processed Save Queue size : ", save_queue.qsize())
                save_queue.put(pipe_data.frame)

        if replay_speed < 1:
            wait_for_ms = 1 + int(2 ** abs(replay_speed - 1) / 10 * 1000)  # magic formula for delay
            frames_to_skip = 0
            key = cv2.waitKey(wait_for_ms)
        else:
            frames_to_skip = replay_speed - 1
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
        elif key & 0xFF == ord('+'):
            replay_speed += 1
            if replay_speed == 0:
                replay_speed = 1
            print(f"Replay speed: {replay_speed}")
        elif key & 0xFF == ord('-'):
            replay_speed -= 1
            if replay_speed == 0:
                replay_speed = -1
            print(f"Replay speed: {replay_speed}")
        timer.stop('Display Frame')

    timer.start('Cleanup', parent='Overall Timer')
    cv2.destroyAllWindows()

    keep_running.value = False
    print("Initiating Termination of MultiProcessingManager")

    if save_queue is not None and save_process is not None:
        print("Joining processed frame saving process")
        save_queue.put("STOP")
        save_process.join()
    print("Processed frame saving process joined")

    print("Joining MultiProcessingManager")
    mp_manager.terminate()
    print("MultiProcessingManager joined")

    timer.stop('Cleanup')
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