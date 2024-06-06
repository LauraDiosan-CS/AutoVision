import argparse
import os
import pickle
import time
from datetime import datetime

import cv2
import ripc

from config import Config
from filters.draw_filter import DrawFilter
from helpers.ControlledProcess import ControlledProcess
from helpers.helpers import stack_images_v2, draw_rois_and_wait, Timer, get_roi_bbox_for_video, save_frames
from ripc import SharedMemoryWriter, SharedMemoryReader
from multiprocessing_manager import MultiProcessingManager
from objects.pipe_data import PipeData
from objects.types.save_info import SaveInfo
from objects.types.video_info import VideoRois, VideoInfo
import multiprocessing as mp


class VideoReaderProcess(ControlledProcess):

    def run(self):
        video_shared_memory = SharedMemoryWriter(name=Config.video_feed_memory_name, size=Config.image_size)
        self.finish_setup()

        fps = Config.fps
        time_between_frames = 1 / fps

        video_path = str(os.path.join(Config.videos_dir, Config.video_name))
        capture = cv2.VideoCapture(video_path)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, Config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.height)
        print(
            f"VideoReaderProcess: Actual video dimensions width: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)} height: {capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        while capture.isOpened():
            start_time = time.perf_counter()

            ret, frame = capture.read()
            if not ret:
                break

            video_shared_memory.write(frame.tobytes())

            end_time = time.perf_counter() - start_time
            time_to_wait = time_between_frames - end_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        capture.release()
        print("VideoReaderProcess: Video ended")


def main():
    mp.set_start_method('spawn')
    # mp.set_sharing_strategy('file_system')

    video_reader_process = VideoReaderProcess()
    video_reader_process.start()
    # ripc.V4lSharedMemoryWriter('path', Config.width, "video_name")

    mp_manager_keep_running = mp.Value('b', True)
    mp_manager = MultiProcessingManager(mp_manager_keep_running)
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
    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)

    while True:
        pipe_data_bytes = visualizer_memory.read()
        if pipe_data_bytes is not None:
            pipe_data: PipeData = pickle.loads(pipe_data_bytes)

            with Timer("Main Process Loop"):
                if Config.apply_visualizer:
                    visualized_frame = draw_filter.process(pipe_data)

                if pipe_data.processed_frames is not None and len(pipe_data.processed_frames) > 0:
                    squashed_frames = sum(pipe_data.processed_frames.values(), [])
                    squashed_frames.append(visualized_frame)
                    final_img = stack_images_v2(1, squashed_frames)
                else:
                    final_img = visualized_frame

                cv2.imshow('CarVision', final_img)

                if Config.save_processed_video and save_queue is not None:
                    print("Processed Save Queue size : ", save_queue.qsize())
                    save_queue.put(visualized_frame)

        if replay_speed < 0:
            wait_for_ms = 1 + int(2 ** abs(replay_speed - 1) / 10 * 1000)  # magic formula for delay
            frames_to_skip = 0
            key = cv2.waitKey(wait_for_ms)
        else:
            frames_to_skip = replay_speed
            key = cv2.waitKey(5)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('x'):
            draw_rois_and_wait(visualized_frame, video_rois)
            cv2.waitKey(0)
        elif key & 0xFF == ord('s'):
            save_dir_path = os.path.join(os.getcwd(), Config.screenshot_dir)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            screenshot_name = f'{Config.video_name}_{timestamp}.jpg'
            screenshot_path = os.path.join(save_dir_path, screenshot_name)
            cv2.imwrite(screenshot_path, visualized_frame)
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

    cv2.destroyAllWindows()

    mp_manager_keep_running.value = False
    print("Initiating Termination of MultiProcessingManager")

    if save_queue is not None and save_process is not None:
        print("Joining processed frame saving process")
        save_queue.put("STOP")
        save_process.join()
    print("Processed frame saving process joined")

    print("Joining MultiProcessingManager")
    mp_manager.terminate()
    print("MultiProcessingManager joined")


def update_config(config, args):
    for arg in vars(args):
        if hasattr(config, arg):
            setattr(config, arg, getattr(args, arg))
    Config.print_config()


if __name__ == '__main__':
    args = argparse.ArgumentParser().parse_args()
    update_config(Config, args)
    main()