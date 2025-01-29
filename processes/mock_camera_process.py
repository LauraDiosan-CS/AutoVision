import multiprocessing as mp
import os
import time
import cv2

from rs_ipc import ReaderWaitPolicy, SharedMessage, OperationMode

from configuration.config import Config, MPStrategy
from processes.controlled_process import ControlledProcess


class MockCameraProcess(ControlledProcess):
    def __init__(
        self,
        start_video: mp.Value,
        keep_running: mp.Value,
        program_start_time: float,
        name: str = None,
    ):
        super().__init__(name=name, program_start_time=program_start_time)
        self.start_video = start_video
        self.keep_running = keep_running

    def run(self):
        try:
            video_feed_shm = SharedMessage.open(
                Config.video_feed_memory_name,
                OperationMode.WriteSync,
                (
                    ReaderWaitPolicy.All()
                    if Config.mp_strategy == MPStrategy.ALL_FRAMES_ALL_PROCESSES
                    else ReaderWaitPolicy.Count(1)
                ),
            )

            time_between_frames = 1 / Config.camera_fps

            video_path = str(os.path.join(Config.videos_dir, Config.video_name))
            capture = cv2.VideoCapture(video_path)

            # Get the actual video dimensions
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Determine if resizing is necessary
            resize_needed = (
                actual_width != Config.width or actual_height != Config.height
            )

            if resize_needed:
                print(
                    f"[CameraProcess] Actual video width: {actual_width} height: {actual_height} so resize is needed"
                )

            while not self.start_video.value and self.keep_running.value:
                pass
            print(
                f"[CameraProcess] Starting video after {(time.perf_counter() - self.program_start_time):.2f} s"
            )

            while self.keep_running.value and capture.isOpened():
                start_time = time.perf_counter()

                ret, frame = capture.read()
                if not ret:
                    print("[CameraProcess] Video ended")
                    break

                # Resize the frame to the desired resolution defined in the Config
                if resize_needed:
                    frame = cv2.resize(
                        frame,
                        (Config.width, Config.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                video_feed_shm.write(frame.tobytes())

                end_time = time.perf_counter() - start_time
                time_to_wait = (
                    time_between_frames - end_time
                )  # make sure video is played at the correct fps
                if time_to_wait > 0:
                    time.sleep(time_to_wait)

            video_feed_shm.stop()
            capture.release()
        except Exception as e:
            print(f"[CameraProcess]: Exception: {e}")
            self.keep_running.value = False