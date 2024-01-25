import argparse

import numpy as np
import yaml
import os
import cv2

from vision_pipeline.pipeline import Pipeline


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def main(args):
    pipeline = Pipeline(args)

    video = os.path.join(args.video_dir, args.video)
    cap = cv2.VideoCapture(video)

    print(video)
    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            processed_frames = pipeline.run_seq(frame)

            if processed_frames is not None:
                imgStack = stackImages(0.5, processed_frames)
                cv2.imshow('CarVision', imgStack)
            else:
                cv2.imshow('CarVision', frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
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