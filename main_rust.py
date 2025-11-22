import multiprocessing as mp
import os
import pickle
import time
import struct
from datetime import datetime

import cv2

import numpy as np
from matplotlib import pyplot as plt
from rs_ipc.rs_ipc import ReaderWaitPolicy

from rs_ipc import SharedMessage, OperationMode
from configuration.config import Config
from perception.helpers import (
    get_roi_bbox_for_video,
    extract_pipeline_names,
    draw_rois_and_wait,
)
from perception.objects.pipe_data import PipeData
from perception.objects.timingvisualizer import TimingVisualizer
from perception.objects.video_info import VideoInfo, VideoRois
from perception.visualize_data import visualize_data
from processes.multiprocessing_manager import MultiProcessingManager

def pack_named_images(items: list[tuple[str, np.ndarray]]) -> bytearray:
    """
    Packs (name, image) tuples into a single bytearray.
    Layout: [64s name][u32 width][u32 height][u32 channels][pixel bytes]...
    """
    # 64s: 64 char bytes, I: uint32
    HEADER_FORMAT = '=64sIII'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    # 1. Pre-calculate total size
    total_size = 0
    for _, img in items:
        total_size += HEADER_SIZE + img.nbytes

    # 2. Allocate single contiguous buffer
    buffer = bytearray(total_size)
    current_offset = 0

    for name, img in items:
        # 3. Analyze dimensions
        height, width = img.shape[0], img.shape[1]

        # Handle Channels: If shape is (H,W), channels=1. If (H,W,C), channels=C
        channels = 1 if img.ndim == 2 else img.shape[2]

        # 4. Prepare Name (Encode -> Truncate -> Pack handles padding)
        name_bytes = name.encode('utf-8')
        # Ensure we don't overflow the 64 byte limit
        if len(name_bytes) > 64:
            name_bytes = name_bytes[:64]

        # 5. Pack Header
        # '64s' automatically pads with null bytes if len < 64
        struct.pack_into(HEADER_FORMAT, buffer, current_offset,
                         name_bytes, width, height, channels)

        current_offset += HEADER_SIZE

        # 6. Copy Pixel Data (Zero-copy from numpy to buffer)
        flat_img = img.ravel()
        data_len = flat_img.nbytes
        buffer[current_offset : current_offset + data_len] = memoryview(flat_img)

        current_offset += data_len

    return buffer

def main():
    program_start_time = time.perf_counter()
    mp.set_start_method("spawn")

    print("[Main] Config:", Config.as_json())

    timing_visualizer = TimingVisualizer()
    ot = "Overall Timer"
    se = "Setup"
    timing_visualizer.start(ot)
    timing_visualizer.start(se, parent=ot)

    recording_dir_path = setup_dir_for_iteration()

    keep_running = mp.Value("b", True)
    start_video = mp.Value("b", False)

    visualization_shm = SharedMessage.create(
        Config.visualization_memory_name,
        Config.max_pipe_data_size,
        OperationMode.ReadSync,
        ReaderWaitPolicy.Count(0)
    )
    rust_ui = SharedMessage.open(
        "rust_ui", OperationMode.WriteAsync
    )

    final_frame_version = mp.Value("i", -1)

    mp_manager = MultiProcessingManager(
        keep_running=keep_running,
        program_start_time=program_start_time,
        start_video=start_video,
        recording_dir_path=recording_dir_path,
        final_frame_version=final_frame_version,
        name="MultiProcessingManager",
    )
    mp_manager.start()

    video_rois: VideoRois = get_roi_bbox_for_video(
        Config.video_name, Config.width, Config.height, Config.roi_config_path
    )
    video_info = VideoInfo(
        video_name=Config.video_name,
        height=Config.height,
        width=Config.width,
        video_rois=video_rois,
    )
    pipeline_names = extract_pipeline_names()

    iteration_counter = 0
    timing_visualizer.stop(se)

    while not rust_ui.is_stopped() and not visualization_shm.is_stopped() and keep_running.value:
        pipe_data_bytes = visualization_shm.read()
        if pipe_data_bytes is None:
            break

        start_time1 = time.perf_counter()
        iteration_counter += 1
        pipe_data: PipeData = pickle.loads(pipe_data_bytes)

        if pipe_data.frame_version == final_frame_version.value:
            print(f"[Main] Received final frame version: {pipe_data.frame_version}")
            break

        dl = f"Data Lifecycle {pipe_data.last_pipeline_name[0]}"
        tf2 = f"Transfer Data (MM -> Viz) {pipe_data.last_pipeline_name[0]}"

        pipe_data.timing_info.stop(tf2)
        pipe_data.timing_info.stop(dl)
        timing_visualizer.timing_info.append_hierarchy(
            pipe_data.timing_info, parent_label_of_other=ot
        )

        drawn_frame = visualize_data(
            video_info=video_info, data=pipe_data, raw_frame=pipe_data.raw_frame
        )
        squashed_frames = [("Main", drawn_frame)]
        if pipe_data.processed_frames is not None and len(pipe_data.processed_frames) > 0:
            for name, pipeline_images in pipe_data.processed_frames.items():
                for (index, image) in enumerate(pipeline_images):
                    squashed_frames.append((f"{name} {index}", image))

        images_bytes = bytes(pack_named_images(squashed_frames))
        # print(time.perf_counter() - start_time1)
        if len(images_bytes) >= rust_ui.payload_max_size():
            print("IMAGES TOOO BIIIIIGGGGGG!!!!!!!!")
        rust_ui.write(images_bytes)


    print(f"[Main] Iteration counter: {iteration_counter}")
    visualization_shm.stop()
    rust_ui.stop()
    keep_running.value = False

    print("[Main] Joining MultiProcessingManager")
    mp_manager.join()
    print("[Main] MultiProcessingManager joined")


def setup_dir_for_iteration():
    os.makedirs(Config.recordings_dir, exist_ok=True)
    video_name = os.path.splitext(Config.video_name)[
        0
    ]  # Extract the name without the extension
    dir_name = f"{video_name}-{Config.camera_fps}FPS-{"VIZ" if Config.enable_pipeline_visualization else "NOVIZ"}-{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}"
    recording_dir_path = os.path.join(Config.recordings_dir, dir_name)
    os.makedirs(recording_dir_path)
    with open(os.path.join(recording_dir_path, "config.json"), "w") as file:
        file.write(Config.as_json())
    return recording_dir_path


if __name__ == "__main__":
    main()

import struct
import numpy as np

