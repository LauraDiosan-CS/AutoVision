import json
import os
import queue
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch.multiprocessing as mp

from config import Config
from filters.base_filter import BaseFilter
from objects.types.pipeline_config_types import FILTER_CLASS_LOOKUP, JSONPipelineConfig, PipelineConfig
from objects.types.save_info import SaveInfo
from objects.types.video_info import VideoInfo, VideoRois

from contextlib import contextmanager


@contextmanager
def Timer(description, min_print_time=0.0):
    """
    Context manager to measure and print the execution time of a code block.

    Parameters:
    - description (str): A description of the code block being timed.
    - min_print_time (float): The minimum elapsed time threshold in seconds
                              for printing the timing information. Defaults to 0.0.

    Usage:
    ```
    with TimedExecution("Code block"):
        # Code block to be timed
    ```
    """
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time > min_print_time:
        print(f"{description}: {elapsed_time} seconds")


@contextmanager
def FixedTimeControl(execution_time, print_if_over):
    """
    Context manager to control the execution time within a fixed limit.

    Parameters:
    - execution_time (float): The maximum allowed execution time in seconds.
    - print_if_over (bool): Flag to print a message if execution time exceeds the limit.

    Usage:
    ```
    with FixedTimeControl(10.0, True):
        # Code block to be executed within the time limit
    ```
    """
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time > execution_time:
        if print_if_over:
            print(f"Execution time {elapsed_time} [s] exceeds "
                  f"the limit of {execution_time} [s]")
        time.sleep(execution_time - elapsed_time)


def parse_pipeline_configuration(JSON_pipeline_config: JSONPipelineConfig, video_info: VideoInfo,
                                 models_dir_path: str) -> PipelineConfig:
    parallel_config: PipelineConfig = []
    for JSON_config in JSON_pipeline_config:
        config: list[BaseFilter] = []
        for filter_class_name, provided_params in JSON_config.items():
            if filter_class_name not in FILTER_CLASS_LOOKUP:
                raise ValueError("Invalid filter name")

            filter_class_with_params = FILTER_CLASS_LOOKUP.get(filter_class_name)
            filter_class = filter_class_with_params.filter_class
            expected_params = filter_class_with_params.expected_params

            # check if the args are valid
            if "model" in provided_params:
                if "model_path" in expected_params:
                    provided_params["model_path"] = os.path.join(models_dir_path, provided_params["model"])
                    del provided_params["model"]
                else:
                    raise ValueError(f"Provided model for {filter_class_name} that does not require it")

            elif ("roi_type" in provided_params and
                  provided_params["roi_type"] not in ["lines", "signs", "traffic_lights", "pedestrians"]):
                raise ValueError(f"Invalid roi_type {provided_params['roi_type']} for {filter_class_name}")

            elif any(arg not in expected_params for arg in provided_params):
                raise ValueError(f"Unexpected argument for {filter_class_name}")

            filter_instance = filter_class(video_info=video_info, **provided_params)
            config.append(filter_instance)
        parallel_config.append(config)

    return parallel_config


def stack_images_v2(scale, imgArray):
    num_images = len(imgArray)

    num_rows = int(np.sqrt(num_images))
    num_cols = int(np.ceil(num_images / num_rows))

    # Resize images based on the specified scale
    resized_images = []
    for img in imgArray:
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to RGB
        resized_img = cv2.resize(img, None, fx=scale, fy=scale)
        resized_images.append(resized_img)

    # Calculate canvas dimensions to fit all images
    canvas_height = num_rows * resized_images[0].shape[0]
    canvas_width = num_cols * resized_images[0].shape[1]

    # Create an empty canvas to place the images
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place resized images onto the canvas
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if idx < num_images:
                y_start = i * resized_images[0].shape[0]
                y_end = (i + 1) * resized_images[0].shape[0]
                x_start = j * resized_images[0].shape[1]
                x_end = (j + 1) * resized_images[0].shape[1]
                canvas[y_start:y_end, x_start:x_end] = resized_images[idx]
                idx += 1

    return canvas


def get_roi_bbox_for_video(video_name, roi_config_path: str) -> VideoRois:
    if not os.path.exists(roi_config_path):
        raise FileNotFoundError(f"File not found: {roi_config_path}")

    with open(roi_config_path, 'r') as file:
        all_video_rois: dict[str, VideoRois] = json.load(file)

    if video_name in all_video_rois.keys():
        return all_video_rois[video_name]
    else:
        raise ValueError(f"Video name {video_name} not found in {roi_config_path}")


def update_roi_bbox_for_video(video_name, roi_bbox, roi_config_path: str):
    if not os.path.exists(roi_config_path):
        raise FileNotFoundError(f"File not found: {roi_config_path}")

    with open(roi_config_path, 'r') as file:
        video_data = json.load(file)

    raise NotImplementedError("This function is not fully implemented yet")


def initialize_config():
    video_rois: VideoRois = get_roi_bbox_for_video(Config.video_name, Config.roi_config_path)
    print("\nVideo ROIs:")
    for roi_type, roi_bbox in video_rois.items():
        print(f"ROI type: {roi_type}, ROI bbox: {roi_bbox}")
    video_info = VideoInfo(video_name=Config.video_name, height=Config.height,
                           width=Config.width, video_rois=video_rois)
    with open(Config.pipeline_config_path, 'r') as f:
        JSON_pipeline_config: JSONPipelineConfig = json.load(f)
    parallel_config = parse_pipeline_configuration(JSON_pipeline_config, video_info, Config.models_dir_path)
    return parallel_config, video_info, video_rois


def draw_rois_and_wait(frame, video_rois):
    for video_roi_bbox in video_rois.values():
        cv2.polylines(frame, np.array([video_roi_bbox]), True, (0, 255, 0), 2)
    imgArr = np.asarray(frame)
    plt.imshow(imgArr)
    plt.show()


def save_frames(save_queue: mp.Queue, save_info: SaveInfo):
    print(f"\nSaving video to: {save_info.video_path}\n")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video_writer = cv2.VideoWriter(
        save_info.video_path,
        fourcc, save_info.fps,
        (save_info.width, save_info.height)
    )

    while True:
        try:
            frame = save_queue.get()
            print("!!! Got a frame !!!", type(frame))
            if isinstance(frame, str) and "STOP" in frame:
                print("Saving process received STOP")
                break
            if isinstance(frame, np.ndarray) and frame.flags.writeable:
                output_video_writer.write(frame)
            else:
                print(f"Invalid frame type for saving: {type(frame)}")
        except queue.Empty:
            pass

    if save_queue.empty():
        print(f"All frames saved")
    else:
        print(f"Saving remaining frames")
    while not save_queue.empty():
        print(f"frames left to save: {save_queue.qsize()}")
        frame = save_queue.get()
        if isinstance(frame, str) and "STOP" in frame:
            continue
        elif isinstance(frame, np.ndarray) and frame.flags.writeable:
            output_video_writer.write(frame)
        else:
            print(f"Invalid frame type for saving: {type(frame)}")

    output_video_writer.release()