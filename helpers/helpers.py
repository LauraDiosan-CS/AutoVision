import json
import os
import queue
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch.multiprocessing as mp

from config import Config
from filters.base_filter import BaseFilter
from objects.types.pipeline_config_types import FILTER_CLASS_LOOKUP, JSONPipelinesTYPE, PipelineConfig
from objects.types.save_info import SaveInfo
from objects.types.video_info import VideoInfo, VideoRois

def parse_pipeline_configuration(JSON_pipelines_config: JSONPipelinesTYPE, video_info: VideoInfo,
                                 models_dir_path: str) -> list[PipelineConfig]:
    pipelines: list[PipelineConfig] = []

    for JSON_pipeline_config in JSON_pipelines_config:
        pipeline_name = JSON_pipeline_config.get("name", "Unnamed Pipeline")
        filters: list[BaseFilter] = []

        filters_config = JSON_pipeline_config.get("filters", {})
        for filter_class_name, provided_params in filters_config.items():
            class_with_expected_params = FILTER_CLASS_LOOKUP.get(filter_class_name)
            if not class_with_expected_params:
                raise ValueError(f"Invalid filter name: {filter_class_name}")

            filter_class = class_with_expected_params.filter_class
            expected_params = class_with_expected_params.expected_params

            # validate the existence of the required files for the models
            if "model" in provided_params:
                if "model_path" in expected_params:
                    provided_params["model_path"] = os.path.join(models_dir_path, provided_params["model"])
                    del provided_params["model"]
                else:
                    raise ValueError(f"Provided model for {filter_class_name} that does not require it")

            # validate the roi_type parameter
            elif ("roi_type" in provided_params and
                  provided_params["roi_type"] not in ["lines", "signs", "traffic_lights", "pedestrians"]):
                raise ValueError(f"Invalid roi_type {provided_params['roi_type']} for {filter_class_name}")

            # validate that the provided parameters are the expected ones
            unexpected_args = [arg for arg in provided_params if arg not in expected_params]
            if unexpected_args:
                raise ValueError(f"Unexpected arguments for {filter_class_name}: {unexpected_args}")

            try:
                filter_instance = filter_class(video_info=video_info, **provided_params)
            except Exception as e:
                raise ValueError(f"Failed to instantiate filter {filter_class_name} with error: {str(e)}")

            filters.append(filter_instance)

        pipelines.append(PipelineConfig(name=pipeline_name, filters=filters))

    return pipelines

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


def initialize_config() -> tuple[list[PipelineConfig], VideoInfo, VideoRois]:
    video_rois: VideoRois = get_roi_bbox_for_video(Config.video_name, Config.roi_config_path)
    print("\nVideo ROIs:")
    for roi_type, roi_bbox in video_rois.items():
        print(f"ROI type: {roi_type}, ROI bbox: {roi_bbox}")
    video_info = VideoInfo(video_name=Config.video_name, height=Config.height,
                           width=Config.width, video_rois=video_rois)
    with open(Config.pipeline_config_path, 'r') as f:
        JSON_pipeline_config: JSONPipelinesTYPE = json.load(f)
    pipelines = parse_pipeline_configuration(JSON_pipeline_config, video_info, Config.models_dir_path)
    return pipelines, video_info, video_rois

def extract_pipeline_names() -> list[str]:
    # Extracts and returns the names of all pipelines from the configuration
    with open(Config.pipeline_config_path, 'r') as f:
        JSON_pipeline_config: JSONPipelinesTYPE = json.load(f)
        if not isinstance(JSON_pipeline_config, list) or not all(isinstance(pipeline, dict) for pipeline in JSON_pipeline_config):
            raise TypeError("JSON_pipeline_config must be a list of dictionaries")

        for pipeline in JSON_pipeline_config:
            if 'name' not in pipeline or not isinstance(pipeline['name'], str):
                raise ValueError("Each pipeline configuration must have a 'name' key of type str")

        return [pipeline["name"] for pipeline in JSON_pipeline_config]


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

def stack_images_v2(scale: float, imgArray: list[np.ndarray]) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Scale must be a positive value.")

    num_images = len(imgArray)
    if num_images == 0:
        raise ValueError("Image array cannot be empty.")

    # Determine number of rows and columns for the grid
    num_rows = int(np.sqrt(num_images))
    num_cols = int(np.ceil(num_images / num_rows))

    # Resize images and determine max width and height
    resized_images = []
    max_width, max_height = 0, 0
    for img in imgArray:
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to RGB
        resized_img = cv2.resize(img, None, fx=scale, fy=scale)
        resized_images.append(resized_img)
        max_height = max(max_height, resized_img.shape[0])
        max_width = max(max_width, resized_img.shape[1])

    # Create an empty canvas to place the images
    canvas_height = num_rows * max_height
    canvas_width = num_cols * max_width
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place resized images onto the canvas
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if idx < num_images:
                y_start = i * max_height
                y_end = y_start + resized_images[idx].shape[0]
                x_start = j * max_width
                x_end = x_start + resized_images[idx].shape[1]
                canvas[y_start:y_end, x_start:x_end] = resized_images[idx]
                idx += 1

    return canvas

def stack_images_v3(scale: float, imgArrayList: list[list[np.ndarray]]) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Scale must be a positive value.")

    if len(imgArrayList) == 0 or any(len(imgArray) == 0 for imgArray in imgArrayList):
        raise ValueError("Image array list cannot be empty, and each image list must have at least one image.")

    # Resize images and determine max width and height across all pipelines
    resized_images_list = []
    max_width, max_height = 0, 0
    for imgArray in imgArrayList:
        resized_images = []
        for img in imgArray:
            if len(img.shape) == 2:  # Grayscale image
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to RGB
            resized_img = cv2.resize(img, None, fx=scale, fy=scale)
            resized_images.append(resized_img)
            max_height = max(max_height, resized_img.shape[0])
            max_width = max(max_width, resized_img.shape[1])
        resized_images_list.append(resized_images)

    # Determine optimal rows and columns to minimize blank space
    total_images = sum(len(pipeline) for pipeline in imgArrayList)
    num_cols = min(max(len(pipeline) for pipeline in imgArrayList),
                   3)  # Limit the number of columns to 3 for better visualization
    num_rows = (total_images + num_cols - 1) // num_cols  # Calculate number of rows needed

    # Create an empty canvas to place the images
    canvas_height = num_rows * max_height
    canvas_width = num_cols * max_width
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place resized images onto the canvas
    current_row, current_col = 0, 0
    for i, resized_images in enumerate(resized_images_list):
        for j, img in enumerate(resized_images):
            y_start = current_row * max_height
            y_end = y_start + img.shape[0]
            x_start = current_col * max_width
            x_end = x_start + img.shape[1]

            # Place the image on the canvas
            canvas[y_start:y_end, x_start:x_start + img.shape[1]] = img

            # Draw index on the top-left corner of each image
            index_text = f"{i},{j}"
            cv2.putText(canvas, index_text, (x_start + 10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # Update current row and column for next image
            current_col += 1
            if current_col >= num_cols:
                current_col = 0
                current_row += 1

    # Draw a border around each grouping of images (pipeline)
    current_row, current_col = 0, 0
    for i, resized_images in enumerate(resized_images_list):
        y_start = current_row * max_height
        y_end = y_start + max_height
        x_start = current_col * max_width
        x_end = x_start + len(resized_images) * max_width
        cv2.rectangle(canvas, (x_start, y_start), (x_end, y_start + max_height), (255, 255, 255), 3)

        current_col += len(resized_images)
        if current_col >= num_cols:
            current_col = 0
            current_row += 1

    return canvas