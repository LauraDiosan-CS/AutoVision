import multiprocessing as mp
import os
import time
from enum import Enum

import cv2

from config import Config
from helpers.controlled_process import ControlledProcess
from ripc import SharedMemoryWriter

class Strategy(Enum):
    LIVE = 1
    ALL_FRAMES_FASTEST_PROCESS = 2
    ALL_FRAMES_ALL_PROCESSES = 3

class VideoReaderProcess(ControlledProcess):
    def __init__(self, start_video: mp.Value, keep_running: mp.Value, last_processed_frame_versions: mp.Array, name=None):
        super().__init__(name=name)
        self.start_video = start_video
        self.keep_running = keep_running
        self.last_processed_frame_versions = last_processed_frame_versions
        self.strategy = Strategy.ALL_FRAMES_FASTEST_PROCESS


    def run(self):
        video_shared_memory = SharedMemoryWriter(name=Config.video_feed_memory_name, size=Config.image_size)
        self.finish_setup()

        time_between_frames = 1 / Config.fps

        video_path = str(os.path.join(Config.videos_dir, Config.video_name))
        capture = cv2.VideoCapture(video_path)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, Config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.height)
        print(
            f"VideoReaderProcess: Actual video dimensions width: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)} height: {capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        while not self.start_video.value and self.keep_running.value:
            pass
        print(f"VideoReaderProcess: Starting video at {(time.perf_counter() - Config.program_start_time):.2f} s")

        while self.keep_running.value and capture.isOpened():
            # print(f"Shared list (version={video_shared_memory.last_written_version()}) : ", end=" ")
            # for last_processed_frame_version in self.last_processed_frame_versions:
            #     print(last_processed_frame_version.value, end=" ")
            # print()

            if self.strategy == Strategy.ALL_FRAMES_ALL_PROCESSES:
                if not all(shared_memory.value == video_shared_memory.last_written_version() for shared_memory in
                           self.last_processed_frame_versions):
                    # print("VideoReaderProcess: Waiting for all processes to catch up")
                    continue
            elif self.strategy == Strategy.ALL_FRAMES_FASTEST_PROCESS:
                if not any(x.value == video_shared_memory.last_written_version() for x in self.last_processed_frame_versions):
                    # print("VideoReaderProcess: Waiting for one process to catch up")
                    continue
            else:
                pass

            start_time = time.perf_counter()

            ret, frame = capture.read()
            if not ret:
                break

            video_shared_memory.write(frame.tobytes())

            end_time = time.perf_counter() - start_time
            time_to_wait = time_between_frames - end_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        video_shared_memory.close()
        capture.release()
        print("VideoReaderProcess: Video ended")