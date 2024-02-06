import argparse
from copy import deepcopy

import numpy as np
import yaml
import os
import cv2
import matplotlib.pyplot as plt
from video_info import VideoInfo
from vision_pipeline.filters.draw_filter import DrawFilter

from vision_pipeline.pipeline import Pipeline
from helpers import get_roi_bbox_for_video, stack_images

def main(args):
    video_path = os.path.join(args.video_dir, args.video)
    cap = cv2.VideoCapture(video_path)

    video_roi_bbox = get_roi_bbox_for_video(args.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_info = VideoInfo(video_name=args.video,video_roi_bbox=video_roi_bbox, height=height, width=width)

    pipeline = Pipeline(args, video_info)


    print(video_path)
    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)

    speed = 1

    while cap.isOpened():
        if speed < 1:
            speed = 1

        for i in range(speed):
            ret, frame = cap.read()

        if ret:
            frame_copy = deepcopy(frame)

            processed_frames, steering_angle, road_markings = pipeline.run_seq(frame)

            draw_filter = DrawFilter(video_info=video_info)

            drawn_frame = draw_filter.process(frame_copy, road_markings, steering_angle)
            
            # processed_frames.append(drawn_frame)
            processed_frames = None

            if processed_frames is not None:
                imgStack = stack_images(0.5, processed_frames)
                cv2.imshow('CarVision', imgStack)
            else:
                cv2.imshow('CarVision', drawn_frame)


            key = cv2.waitKey(5)
            # Press Q on keyboard to exit
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('w'):
                speed += 1
            elif key & 0xFF == ord('e'):
                speed -= 1
            elif key & 0xFF == ord('x'):
                cv2.polylines(frame, np.array([video_roi_bbox]), True, (0, 255, 0), 2)
                imgArr = np.asarray(frame.copy())
                plt.imshow(imgArr)
                plt.show()
                cv2.waitKey(0)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_config(parser):
    config_file = os.path.join(os.path.dirname(__file__), os.path.join('vision_pipeline', 'config.yaml'))

    with open(config_file, 'r') as config:
        data = yaml.safe_load(config)

        for arg in data.keys():
            # Check if the value is a path before using os.path.join
            if os.path.sep in str(data[arg]):
                data[arg] = os.path.join(os.path.dirname(__file__), data[arg])
            print(f'--{arg} = {data[arg]}')
            parser.add_argument(f'--{arg}', default=data[arg])
        args = parser.parse_args()

    config.close()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_config(parser)
    main(args)