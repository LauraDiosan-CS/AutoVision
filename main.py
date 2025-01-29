import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt
from rs_ipc import SharedMessage, OperationMode, ReaderWaitPolicy
from configuration.config import Config
from perception.helpers import (
    get_roi_bbox_for_video,
    extract_pipeline_names,
    draw_rois_and_wait,
    stack_images_v4,
)
from perception.objects.pipe_data import PipeData
from perception.objects.timingvisualizer import TimingVisualizer
from perception.objects.video_info import VideoInfo, VideoRois
from perception.visualize_data import visualize_data
from processes.multiprocessing_manager import MultiProcessingManager


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
    )
    mp_manager = MultiProcessingManager(
        keep_running=keep_running,
        program_start_time=program_start_time,
        start_video=start_video,
        recording_dir_path=recording_dir_path,
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

    pipe_data = None
    iteration_counter = 0
    cv2.namedWindow("CarVision", cv2.WINDOW_NORMAL)
    timing_visualizer.stop(se)

    while True:
        pipe_data_bytes = visualization_shm.read(block=False)

        if pipe_data_bytes is not None:
            iteration_counter += 1
            pipe_data: PipeData = pickle.loads(pipe_data_bytes)

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
            if (
                pipe_data.processed_frames is not None
                and len(pipe_data.processed_frames) > 0
            ):
                squashed_frames = [[drawn_frame]]

                for pipeline_name in pipeline_names:
                    if pipeline_name in pipe_data.processed_frames:
                        squashed_frames.append(
                            pipe_data.processed_frames[pipeline_name]
                        )
                    else:
                        # add black frame
                        squashed_frames.append(
                            [np.zeros((Config.height, Config.width, 3), dtype=np.uint8)]
                        )
                stacked_frame = stack_images_v4(1, squashed_frames)
                # print(f"Stacking took: {(time.perf_counter() - start_time) * 1000:.2f}ms")
                cv2.imshow("CarVision", stacked_frame)
            else:
                cv2.imshow("CarVision", drawn_frame)

        key = cv2.waitKey(5)
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("x"):
            if pipe_data is not None:
                draw_rois_and_wait(pipe_data.frame, video_rois)
                cv2.waitKey(0)
            else:
                print("No frame to draw ROIs on")

    timing_visualizer.start("Cleanup", parent=ot)
    cv2.destroyAllWindows()

    visualization_shm.stop()
    keep_running.value = False

    print("[Main] Joining MultiProcessingManager")
    mp_manager.join()
    print("[Main] MultiProcessingManager joined")

    timing_visualizer.stop("Cleanup")
    timing_visualizer.stop(ot)
    print(timing_visualizer.timing_info)
    timing_visualizer.plot_pie_charts(
        save_path=os.path.join(recording_dir_path, "timings")
    )
    plt.show()  # Keep the pie chart open


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