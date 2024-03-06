import argparse
import os
import time

import cv2
import torch.multiprocessing as mp
from config import Config
from helpers.helpers import stack_images_v2, initialize_config, draw_rois_and_wait, Timer
from multiprocessing_manager import MultiProcessingManager



def camera_process(queue):
    video_path = str(os.path.join(Config.videos_dir, Config.video_name))
    cap = cv2.VideoCapture(video_path)
    Config.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Config.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        with Timer("Camera Process Loop", min_print_time=0.01):
            ret, frame = cap.read()
            if ret:
                if not queue.full():
                    queue.put(frame)
            else:
                break

    cap.release()


def main():
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')

    queue = mp.Queue()

    camera_proc = mp.Process(target=camera_process, args=(queue,))
    camera_proc.start()

    parallel_config, video_info, video_rois = initialize_config()

    mp_manager = MultiProcessingManager(parallel_config, video_info, save_output=True)
    #TODO move save input into new process maybe move process into manager

    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)

    replay_speed = 1
    frames_to_skip = 0

    while True:
        if not queue.empty():
            frame = queue.get()
        else:
            continue
        with Timer("Main Process Loop"):
            print()
            with Timer("MP Manager"):
                data = mp_manager.process_frame(frame, apply_draw_filter=True)

            with Timer("CV2 Show"):
                if Config.visualize_only_final:
                    cv2.imshow('CarVision', data.frame)
                elif data.processed_frames is not None and len(data.processed_frames) > 0:
                    imgStack = stack_images_v2(1, data.processed_frames)
                    cv2.imshow('CarVision', imgStack)

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
                draw_rois_and_wait(frame, video_rois)
                cv2.waitKey(0)
            elif key & 0xFF == ord('h'):
                pass
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

    mp_manager.finish_saving()

    camera_proc.join()
    cv2.destroyAllWindows()


def update_config(config, args):
    for arg in vars(args):
        if hasattr(config, arg):
            setattr(config, arg, getattr(args, arg))
    Config.print_config()


if __name__ == '__main__':
    args = argparse.ArgumentParser().parse_args()
    update_config(Config, args)
    main()