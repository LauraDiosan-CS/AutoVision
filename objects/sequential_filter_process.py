import pickle
import time
import numpy as np

import multiprocessing as mp

from filters.base_filter import BaseFilter
from ripc import SharedMemoryReader, SharedMemoryWriter
from config import Config
from helpers.ControlledProcess import ControlledProcess
from objects.pipe_data import PipeData


class SequentialFilterProcess(ControlledProcess):
    __slots__ = ['filter_configuration', 'keep_running']

    def __init__(self, filter_configuration: list[BaseFilter], process_name: str, keep_running: mp.Value):
        super().__init__(name=process_name)
        self.filter_configuration = filter_configuration
        self.keep_running = keep_running

    def run(self):
        memory_writer = SharedMemoryWriter(name=self.name, size=Config.pipe_memory_size)
        self.finish_setup()
        video_feed = SharedMemoryReader(name=Config.video_feed_memory_name)

        while self.keep_running:
            frame_as_bytes = video_feed.read_in_place(ignore_same_version=True)
            while frame_as_bytes is None:
                if not self.keep_running.value:
                    print(f"Exiting {self.name}")
                    return
                time.sleep(0.01)
                frame_as_bytes = video_feed.read_in_place(ignore_same_version=True)

            start_time = time.time()
            frame = np.frombuffer(frame_as_bytes, dtype=np.uint8).reshape((Config.height, Config.width, 3))

            data = PipeData(frame=frame,
                            depth_frame=None,
                            unfiltered_frame=frame)

            data.last_touched_process = self.name
            for filter in self.filter_configuration:
                data = filter.process(data)

            data_as_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            end_time = time.time()
            data.pipeline_execution_time = end_time - start_time

            memory_writer.write(data_as_bytes)

        print(f"Exiting {self.name}")