import multiprocessing as mp
import os
import time
from datetime import datetime

from configuration.config import Config
from processes.multiprocessing_manager_rust_ui import MultiProcessingManager

def main():
    program_start_time = time.perf_counter()
    mp.set_start_method("spawn")
    # print("[Main] Config:", Config.as_json())

    recording_dir_path = setup_dir_for_iteration()

    keep_running = mp.Value("b", True)
    start_video = mp.Value("b", False)

    final_frame_version = mp.Value("i", -1)

    mp_manager = MultiProcessingManager(
        keep_running=keep_running,
        program_start_time=program_start_time,
        start_video=start_video,
        recording_dir_path=recording_dir_path,
        final_frame_version=final_frame_version,
        name="MultiProcessingManager",
    )
    mp_manager.run()

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