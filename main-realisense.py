import argparse
import numpy as np
import os
import cv2
import torch.multiprocessing as mp

from helpers.helpers import parse_config, stack_images_v2, initialize_config, draw_rois_and_wait
from process_pipeline_manager import ProcessPipelineManager

import pyrealsense2 as rs


def main(args):
    mp.set_start_method('spawn')

    pipelineCamera = rs.pipeline()
    realsense_config = rs.config()
    realsense_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.fps)
    realsense_config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipelineCamera.start(realsense_config)

    parallel_config, video_info, video_rois = initialize_config(args)

    pool_manager = ProcessPipelineManager(parallel_config, video_info)

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

        data = pool_manager.run(color_frame, depth_frame)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    args = parse_config(parser, config_file_path)
    main(args)