import multiprocessing as mp
import os
import time
from enum import Enum

import cv2

from ripc import SharedMemoryWriter

from configuration.config import Config
from processes.controlled_process import ControlledProcess


class Strategy(Enum):
    LIVE = 1
    ALL_FRAMES_FASTEST_PROCESS = 2
    ALL_FRAMES_ALL_PROCESSES = 3

class MockCameraProcess(ControlledProcess):
    def __init__(self, start_video: mp.Value, keep_running: mp.Value, last_processed_frame_versions: mp.Array, program_start_time: float, name: str = None):
        super().__init__(name=name, program_start_time=program_start_time)
        self.start_video = start_video
        self.keep_running = keep_running
        self.last_processed_frame_versions = last_processed_frame_versions
        self.strategy = Strategy.ALL_FRAMES_FASTEST_PROCESS


    def run(self):
        camera_shared_memory = SharedMemoryWriter(name=Config.video_feed_memory_name, size=Config.frame_size)
        self.finish_setup()

        time_between_frames = 1 / Config.fps

        video_path = str(os.path.join(Config.videos_dir, Config.video_name))
        capture = cv2.VideoCapture(video_path)

        # Get the actual video dimensions
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine if resizing is necessary
        resize_needed = actual_width != Config.width or actual_height != Config.height

        print(f"MockCameraProcess: Actual video width: {actual_width} height: {actual_height}, resize_needed: {resize_needed}")

        while not self.start_video.value and self.keep_running.value:
            pass
        print(f"MockCameraProcess: Starting video at {(time.perf_counter() - self.program_start_time):.2f} s")

        while self.keep_running.value and capture.isOpened():
            start_time = time.perf_counter()

            if self.strategy == Strategy.ALL_FRAMES_ALL_PROCESSES:
                if not all(shared_memory.value == camera_shared_memory.last_written_version() for shared_memory in
                           self.last_processed_frame_versions):
                    # print("CameraProcess: Waiting for all processes to catch up")
                    continue
            elif self.strategy == Strategy.ALL_FRAMES_FASTEST_PROCESS:
                if not any(x.value == camera_shared_memory.last_written_version() for x in self.last_processed_frame_versions):
                    # print("CameraProcess: Waiting for one process to catch up")
                    continue
            else:
                print("CameraProcess: Invalid strategy")
                pass

            ret, frame = capture.read()
            if not ret:
                print("MockCameraProcess: Video ended")
                break

            # Resize the frame to the desired resolution defined in the Config
            if resize_needed:
                frame = cv2.resize(frame, (Config.width, Config.height), interpolation=cv2.INTER_LINEAR)

            camera_shared_memory.write(frame.tobytes())

            end_time = time.perf_counter() - start_time
            time_to_wait = time_between_frames - end_time # make sure video is played at the correct fps
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        camera_shared_memory.close()
        capture.release()
        print("MockCameraProcess: Video ended")