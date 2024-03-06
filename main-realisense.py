import argparse
import numpy as np
import cv2
import torch.multiprocessing as mp

from config import Config
from helpers.helpers import stack_images_v2, initialize_config, draw_rois_and_wait
from multiprocessing_manager import MultiProcessingManager

import pyrealsense2 as rs


def camera_process(queue):
    pipelineCamera = rs.pipeline()
    realsense_config = rs.config()
    realsense_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, Config.fps)
    realsense_config.enable_stream(rs.stream.color, Config.width, Config.height, rs.format.bgr8, Config.fps)
    pipelineCamera.start(realsense_config)

    while True:
        frames = pipelineCamera.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame and not depth_frame:
            print("!!! No frames received from camera !!!")
            continue

        color_frame = np.asanyarray(color_frame.get_data())
        depth_frame = np.asanyarray(depth_frame.get_data())

        queue.put((color_frame, depth_frame))

def main():
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')

    pipelineCamera = rs.pipeline()
    realsense_config = rs.config()
    realsense_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, Config.fps)
    realsense_config.enable_stream(rs.stream.color, Config.width, Config.height, rs.format.bgr8, Config.fps)
    pipelineCamera.start(realsense_config)

    parallel_config, video_info, video_rois = initialize_config()

    pool_manager = MultiProcessingManager(parallel_config, video_info, save_input=True)

    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)

    while True:
        frames = pipelineCamera.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame and not depth_frame:
            print("!!! No frames received from camera !!!")
            continue

        color_frame = np.asanyarray(color_frame.get_data())
        depth_frame = np.asanyarray(depth_frame.get_data())

        data = pool_manager.process_frame(color_frame, depth_frame, apply_draw_filter=True)

        if data.processed_frames is not None and len(data.processed_frames) > 0:
            imgStack = stack_images_v2(1, data.processed_frames)
            cv2.imshow('CarVision', imgStack)
        else:
            cv2.imshow('CarVision', data.frame)

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('x'):
            draw_rois_and_wait(color_frame, video_rois)
            cv2.waitKey(0)
        elif key & 0xFF == ord('s'):
            pool_manager.finish_saving()

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