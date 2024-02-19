import argparse
import json
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from helpers.helpers import get_roi_bbox_for_video, parse_pipeline_configuration, parse_config, stack_images_v2
from objects.pipe_data import PipeData
from objects.types.pipeline_config_types import JSONPipelineConfig
from objects.types.video_info import VideoRois, VideoInfo
from process_pipeline_manager import ProcessPipelineManager


def main(args):
    mp.set_start_method('spawn')

    video_path = os.path.join(args.videos_dir, args.video_name)

    cap = cv2.VideoCapture(video_path)

    video_rois: VideoRois = get_roi_bbox_for_video(args.video_name, args.roi_config_path)
    print("Video ROIs:")
    for roi_type, roi_bbox in video_rois.items():
        print(f"ROI type: {roi_type}, ROI bbox: {roi_bbox}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_info = VideoInfo(video_name=args.video_name, height=height,
                           width=width, video_rois=video_rois)

    with open(args.pipeline_config_path, 'r') as f:
        JSON_pipeline_config: JSONPipelineConfig = json.load(f)

    parallel_config = parse_pipeline_configuration(JSON_pipeline_config, video_info, args.models_dir_path)

    pool_manager = ProcessPipelineManager(parallel_config, video_info)

    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)

    replay_speed = 1
    ret, frame = None, None
    while cap.isOpened():
        if replay_speed < 1:
            replay_speed = 1

        for i in range(replay_speed):
            ret, frame = cap.read()

        if ret:
            data: PipeData = pool_manager.run(frame.copy())

            if data.processed_frames is not None and len(data.processed_frames) > 0:
                imgStack = stack_images_v2(1, data.processed_frames)
                cv2.imshow('CarVision', imgStack)
            else:
                cv2.imshow('CarVision', data.frame)

            key = cv2.waitKey(5)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('w'):
                replay_speed += 1
            elif key & 0xFF == ord('e'):
                replay_speed -= 1
            elif key & 0xFF == ord('x'):
                for video_roi_bbox in video_rois.values():
                    cv2.polylines(frame, np.array([video_roi_bbox]), True, (0, 255, 0), 2)
                imgArr = np.asarray(frame.copy())
                plt.imshow(imgArr)
                plt.show()
                cv2.waitKey(0)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    args = parse_config(parser, config_file_path)
    main(args)