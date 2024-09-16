import pickle
import time
import numpy as np

import multiprocessing as mp

from filters.base_filter import BaseFilter
from ripc import SharedMemoryReader, SharedMemoryWriter
from config import Config
from helpers.controlled_process import ControlledProcess
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

        while self.keep_running.value:
            frame_as_bytes = video_feed.blocking_read()
            if frame_as_bytes is None:
                break

            frame = np.frombuffer(frame_as_bytes, dtype=np.uint8).reshape((Config.height, Config.width, 3))

            start_time = time.time()
            data = PipeData(frame=frame,
                            depth_frame=None,
                            unfiltered_frame=frame, creation_time=time.time())
            data.last_touched_process = self.name
            data.timings.start("Data Lifecycle")

            data.timings.start("Process Data", parent="Data Lifecycle")

            for filter in self.filter_configuration:
                filter.process(data)

            end_time = time.time()
            data.pipeline_execution_time = end_time - start_time
            data.timings.stop("Process Data")

            data.send_start_time = time.time()
            data.timings.start("Transfer Data (SPF -> MM)", parent="Data Lifecycle")

            data_as_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            memory_writer.write(data_as_bytes)

        memory_writer.close()
        print(f"Exiting {self.name}")