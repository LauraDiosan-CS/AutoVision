import multiprocessing as mp
import os
import time

import cv2

from config import Config
from helpers.controlled_process import ControlledProcess
from ripc import SharedMemoryWriter


class VideoReaderProcess(ControlledProcess):
    def __init__(self, keep_running: mp.Value, name=None):
        super().__init__(name=name)
        self.keep_running = keep_running

    def run(self):
        video_shared_memory = SharedMemoryWriter(name=Config.video_feed_memory_name, size=Config.image_size)
        self.finish_setup()

        fps = Config.fps
        time_between_frames = 1 / fps

        video_path = str(os.path.join(Config.videos_dir, Config.video_name))
        capture = cv2.VideoCapture(video_path)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, Config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.height)
        print(
            f"VideoReaderProcess: Actual video dimensions width: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)} height: {capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        while self.keep_running.value and capture.isOpened():
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