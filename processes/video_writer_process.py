from copy import deepcopy

import cv2
import numpy as np
from ripc import SharedMemoryCircularQueue
import multiprocessing as mp

from configuration.config import Config, VisualizationStrategy
from perception.objects.save_info import SaveInfo
from processes.controlled_process import ControlledProcess


class VideoWriterProcess(ControlledProcess):
    def __init__(self, save_info: SaveInfo, shared_memory_name: str, keep_running: mp.Value, program_start_time: float, name: str = None):
        super().__init__(name=name, program_start_time=program_start_time)
        self.save_info = save_info
        self.shared_memory_name = shared_memory_name
        self.keep_running = keep_running
        self.program_start_time = program_start_time

    def run(self):
        save_queue: SharedMemoryCircularQueue = SharedMemoryCircularQueue.open(self.shared_memory_name)

        self.finish_setup()

        video_writer = cv2.VideoWriter(self.save_info.video_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       self.save_info.fps,
                                       (self.save_info.width, self.save_info.height))

        frame_as_bytes = None
        cv2.namedWindow("Saved Video", cv2.WINDOW_NORMAL)
        while self.keep_running.value:
            if Config.visualizer_strategy == VisualizationStrategy.NEWEST_FRAME:
                if len(save_queue) == 0:
                    frame_as_bytes = None
                else:
                    list_pipe_data_bytes = save_queue.read_all()
                    frame_as_bytes = list_pipe_data_bytes[-1]
            elif Config.visualizer_strategy == VisualizationStrategy.ALL_FRAMES:
                frame_as_bytes = save_queue.try_read()

            if frame_as_bytes is None:
                continue

            frame = np.frombuffer(frame_as_bytes, dtype=np.uint8).reshape((Config.height, Config.width, 3))
            cv2.imshow("Saved Video", frame)
            cv2.waitKey(1)

            video_writer.write(frame)

        print("VideoWriterProcess: Video ended")
        video_writer.release()