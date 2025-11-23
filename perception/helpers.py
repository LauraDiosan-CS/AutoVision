import json
import os
import struct

import cv2
import numpy as np
from matplotlib import pyplot as plt

from configuration.config import Config
from perception.filters.base_filter import BaseFilter
from perception.objects.pipeline_config_types import PipelineConfig, JSONPipelinesTYPE, FILTER_CLASS_LOOKUP
from perception.objects.video_info import VideoInfo, VideoRois

def pack_named_images(description: str, items: list[tuple[str, np.ndarray]]) -> bytearray:
    """
    Packs (name, image) tuples into a single bytearray.
    Layout: [55s name][u32 width][u32 height][u8 channels][pixel bytes]...
    """
    DESCRIPTION_SIZE = 1024
    NAME_SIZE = 55
    HEADER_FORMAT = f'={NAME_SIZE}sIIB'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT) # 64 bytes

    # 1. Pre-calculate total size
    total_size = DESCRIPTION_SIZE
    for _, img in items:
        total_size += HEADER_SIZE + img.nbytes

    # 2. Allocate single contiguous buffer
    buffer = bytearray(total_size)

    # Write description
    description_bytes = description.encode('utf-8')
    # Ensure we don't overflow the DESCRIPTION_SIZE byte limit
    if len(description_bytes) > DESCRIPTION_SIZE:
        description_bytes = description_bytes[:DESCRIPTION_SIZE]
    buffer[:len(description_bytes)] = description_bytes

    current_offset = DESCRIPTION_SIZE

    for name, img in items:
        # 3. Analyze dimensions
        height, width = img.shape[0], img.shape[1]

        # Handle Channels: If shape is (H,W), channels=1. If (H,W,C), channels=C
        channels = 1 if img.ndim == 2 else img.shape[2]

        # 4. Prepare Name (Encode -> Truncate -> Pack handles padding)
        name_bytes = name.encode('utf-8')
        # Ensure we don't overflow the NAME_SIZE byte limit
        if len(name_bytes) > NAME_SIZE:
            name_bytes = name_bytes[:NAME_SIZE]

        # 5. Pack Header
        # '55s' automatically pads with null bytes if len < NAME_SIZE
        struct.pack_into(HEADER_FORMAT, buffer, current_offset,
                         name_bytes, width, height, channels)

        current_offset += HEADER_SIZE

        # 6. Copy Pixel Data (Zero-copy from numpy to buffer)
        flat_img = img.ravel()
        data_len = flat_img.nbytes
        buffer[current_offset : current_offset + data_len] = memoryview(flat_img)

        current_offset += data_len

    return buffer

def parse_pipeline_configuration(JSON_pipelines_config: JSONPipelinesTYPE, video_info: VideoInfo,
                                 models_dir_path: str, enable_pipeline_visualization: bool = True) -> list[PipelineConfig]:
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

            if "visualize" in provided_params and not enable_pipeline_visualization:
                provided_params["visualize"] = False # Disable visualization if not enabled in config

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


def get_roi_bbox_for_video(video_name, video_width, video_height, roi_config_path: str) -> VideoRois:
    if not os.path.exists(roi_config_path):
        raise FileNotFoundError(f"File not found: {roi_config_path}")

    with open(roi_config_path, 'r') as file:
        all_video_rois: dict[str, VideoRois] = json.load(file)

    # Check if video name exists in the configuration
    if video_name in all_video_rois.keys():
        return all_video_rois[video_name]

    # Fallback to default based on video resolution
    resolution_key = f"{video_width}x{video_height}"
    if resolution_key in all_video_rois.keys():
        return all_video_rois[resolution_key]
    else:
        raise ValueError(
            f"Video name {video_name} not found and no default resolution for {resolution_key} in {roi_config_path}")

def initialize_config(enable_pipeline_visualization: bool) -> tuple[list[PipelineConfig], VideoInfo, VideoRois]:
    video_rois: VideoRois = get_roi_bbox_for_video(Config.video_name, Config.width, Config.height, Config.roi_config_path)

    video_info = VideoInfo(video_name=Config.video_name, height=Config.height,
                           width=Config.width, video_rois=video_rois)
    with open(Config.pipeline_config_path, 'r') as f:
        JSON_pipeline_config: JSONPipelinesTYPE = json.load(f)
    pipelines = parse_pipeline_configuration(JSON_pipeline_config, video_info, Config.models_dir_path, enable_pipeline_visualization)
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

def stack_images_v2(scale, img_array_list):
    num_images = len(img_array_list)

    num_rows = int(np.sqrt(num_images))
    num_cols = int(np.ceil(num_images / num_rows))

    # Resize images based on the specified scale
    resized_images = []
    for img in img_array_list:
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

def stack_images_v2(scale: float, img_array_list: list[np.ndarray]) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Scale must be a positive value.")

    num_images = len(img_array_list)
    if num_images == 0:
        raise ValueError("Image array cannot be empty.")

    # Determine number of rows and columns for the grid
    num_rows = int(np.sqrt(num_images))
    num_cols = int(np.ceil(num_images / num_rows))

    # Resize images and determine max width and height
    resized_images = []
    max_width, max_height = 0, 0
    for img in img_array_list:
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

def stack_images_v3(scale: float, img_array_list: list[list[np.ndarray]]) -> np.ndarray:
    """
    Stacks and arranges multiple images into a single image canvas.

    This function takes a list of lists of images, resizes them according to the given scale factor,
    and arranges them into a grid layout on a single canvas image. Each sublist in the input represents
    a group (pipeline) of images that will be grouped together on the canvas. The function also labels
    each image with its indices and draws borders around each group of images.

    Parameters
    ----------
    scale : float
        Scaling factor to resize the images. Must be a positive value (> 0).
    img_array_list : list of list of numpy.ndarray
        A list containing sublists of images. Each sublist represents a group of images (pipeline),
        and each image is a numpy ndarray.

    Returns
    -------
    numpy.ndarray
        A single image canvas with all the images arranged in a grid, grouped, and labeled.

    Raises
    ------
    ValueError
        If `scale` is not a positive value.
        If `imgArrayList` is empty or any sublist in `imgArrayList` is empty.

    Notes
    -----
    - Images are resized according to the `scale` parameter.
    - Grayscale images are converted to BGR color images.
    - Images are arranged in a grid layout, with a maximum of 3 columns for better visualization.
    - Each group of images (pipeline) is enclosed within a border on the canvas.
    - Each image is labeled with its indices (i, j) on the top-left corner.

    Examples
    --------
    >>> import cv2
    >>> img1 = cv2.imread('image1.jpg')
    >>> img2 = cv2.imread('image2.jpg')
    >>> img3 = cv2.imread('image3.jpg')
    >>> imgArrayList = [[img1, img2], [img3]]
    >>> canvas = stack_images_v3(0.5, img_array_list)
    >>> cv2.imshow('Canvas', canvas)
    >>> cv2.waitKey(0)
    """
    if scale <= 0:
        raise ValueError("Scale must be a positive value.")

    if len(img_array_list) == 0 or any(len(imgArray) == 0 for imgArray in img_array_list):
        raise ValueError("Image array list cannot be empty, and each image list must have at least one image.")

    # Resize images and determine max width and height across all pipelines
    resized_images_list = []
    max_width, max_height = 0, 0
    for imgArray in img_array_list:
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
    total_images = sum(len(pipeline) for pipeline in img_array_list)
    num_cols = min(max(len(pipeline) for pipeline in img_array_list),
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
            canvas[y_start:y_end, x_start:x_end] = img

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
        cv2.rectangle(canvas, (x_start, y_start), (x_end, y_end), (255, 255, 255), 3)

        current_col += len(resized_images)
        if current_col >= num_cols:
            current_col = 0
            current_row += 1

    return canvas

def stack_images_v4(scale: float, img_array_list: list[list[np.ndarray]]) -> np.ndarray:
    """
    Stacks and arranges multiple images into a single image canvas.

    This function takes a list of lists of images, resizes them according to the given scale factor,
    and arranges them into a grid layout on a single canvas image. Each sublist in the input represents
    a group (pipeline) of images that will be grouped together on the canvas. The function also labels
    each image with its indices and draws borders around each group of images.

    Parameters
    ----------
    scale : float
        Scaling factor to resize the images. Must be a positive value (> 0).
    img_array_list : list of list of numpy.ndarray
        A list containing sublists of images. Each sublist represents a group of images (pipeline),
        and each image is a numpy ndarray.

    Returns
    -------
    numpy.ndarray
        A single image canvas with all the images arranged in a grid, grouped, and labeled.

    Raises
    ------
    ValueError
        If `scale` is not a positive value.
        If `imgArrayList` is empty or any sublist in `imgArrayList` is empty.

    Notes
    -----
    - Images are resized according to the `scale` parameter.
    - Grayscale images are converted to BGR color images.
    - All images are resized to match the largest width and height after scaling.
    - Images are arranged in a grid layout, with a maximum of 3 columns for better visualization.
    - Each group of images (pipeline) is enclosed within a border on the canvas.
    - Each image is labeled with its indices (i, j) on the top-left corner.

    Examples
    --------
    >>> import cv2
    >>> img1 = cv2.imread('image1.jpg')
    >>> img2 = cv2.imread('image2.jpg')
    >>> img3 = cv2.imread('image3.jpg')
    >>> imgArrayList = [[img1, img2], [img3]]
    >>> canvas = stack_images_v3(0.5, img_array_list)
    >>> cv2.imshow('Canvas', canvas)
    >>> cv2.waitKey(0)
    """
    if scale <= 0:
        raise ValueError("Scale must be a positive value.")

    if not img_array_list or any(not imgArray for imgArray in img_array_list):
        raise ValueError("Image array list cannot be empty, and each image list must have at least one image.")

    # Step 1: Resize images based on scale and ensure all images are in BGR format
    resized_images_list = []
    for pipeline_idx, imgArray in enumerate(img_array_list):
        resized_pipeline = []
        for img_idx, img in enumerate(imgArray):
            if len(img.shape) == 2:  # Grayscale image
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            resized_pipeline.append(resized_img)
        resized_images_list.append(resized_pipeline)

    # Step 2: Determine the maximum width and height among all resized images
    max_width = max(img.shape[1] for pipeline in resized_images_list for img in pipeline)
    max_height = max(img.shape[0] for pipeline in resized_images_list for img in pipeline)

    # Step 3: Resize all images to match the maximum dimensions
    standardized_images_list = []
    for pipeline in resized_images_list:
        standardized_pipeline = []
        for img in pipeline:
            if img.shape[1] != max_width or img.shape[0] != max_height:
                # Option 1: Direct resize (may distort aspect ratio)
                # standardized_img = cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_AREA)

                # Option 2: Pad the image to maintain aspect ratio
                standardized_img = pad_image_to_size(img, max_width, max_height)
                standardized_pipeline.append(standardized_img)
            else:
                standardized_pipeline.append(img)
        standardized_images_list.append(standardized_pipeline)

    # Step 4: Determine grid size
    max_cols = 3  # Maximum number of columns
    grid_cols = min(max(len(pipeline) for pipeline in standardized_images_list), max_cols)
    grid_rows = (sum(len(pipeline) for pipeline in standardized_images_list) + grid_cols - 1) // grid_cols

    # Step 5: Create canvas
    canvas_height = grid_rows * max_height + (grid_rows + 1) * 10  # Adding 10px padding between rows
    canvas_width = grid_cols * max_width + (grid_cols + 1) * 10   # Adding 10px padding between columns
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 50  # Dark gray background

    # Step 6: Place images on the canvas
    current_row = 0
    current_col = 0
    image_positions = []  # To keep track of image positions for drawing borders

    for pipeline_idx, pipeline in enumerate(standardized_images_list):
        for img_idx, img in enumerate(pipeline):
            y_start = current_row * (max_height + 10) + 10
            y_end = y_start + max_height
            x_start = current_col * (max_width + 10) + 10
            x_end = x_start + max_width

            # Place the image on the canvas
            canvas[y_start:y_end, x_start:x_end] = img

            # Draw index on the top-left corner of each image
            index_text = f"{pipeline_idx},{img_idx}"
            cv2.putText(
                canvas, index_text, (x_start + 10, y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Record the position for border drawing
            image_positions.append((x_start, y_start, x_end, y_end))

            # Update column and row
            current_col += 1
            if current_col >= grid_cols:
                current_col = 0
                current_row += 1

    # Step 7: Draw borders around each pipeline group
    # Calculate starting and ending indices for each pipeline in image_positions
    img_counter = 0
    for pipeline_idx, pipeline in enumerate(standardized_images_list):
        num_imgs_in_pipeline = len(pipeline)
        if num_imgs_in_pipeline == 0:
            continue
        # Get positions of the first and last image in the pipeline
        first_img_pos = image_positions[img_counter]
        last_img_pos = image_positions[img_counter + num_imgs_in_pipeline - 1]
        img_counter += num_imgs_in_pipeline

        # Calculate the bounding box
        top_left = (first_img_pos[0] - 5, first_img_pos[1] - 5)
        bottom_right = (last_img_pos[2] + 5, last_img_pos[3] + 5)
        cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), 3)  # Green border

    return canvas

def pad_image_to_size(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Pads the image to the target width and height while maintaining aspect ratio.

    Parameters
    ----------
    img : numpy.ndarray
        The input image to pad.
    target_width : int
        The desired width after padding.
    target_height : int
        The desired height after padding.

    Returns
    -------
    numpy.ndarray
        The padded image with dimensions (target_height, target_width, 3).
    """
    height, width = img.shape[:2]
    delta_w = target_width - width
    delta_h = target_height - height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [50, 50, 50]  # Dark gray padding
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_img