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

    def __init__(self, filter_configuration: list[BaseFilter], keep_running: mp.Value, processed_frame_count: mp.Value, process_name):
        super().__init__(name=process_name)
        self.filter_configuration = filter_configuration
        self.keep_running = keep_running
        self.processed_frame_count = processed_frame_count

    def run(self):
        memory_writer = SharedMemoryWriter(name=self.name, size=Config.pipe_memory_size)
        self.finish_setup()
        video_feed = SharedMemoryReader(name=Config.video_feed_memory_name)

        while self.keep_running.value:
            frame_as_bytes = video_feed.blocking_read()
            print(f"Received frame in {self.name}")

            if frame_as_bytes is None:
                break

            self.processed_frame_count.value = video_feed.last_read_version()

            frame = np.frombuffer(frame_as_bytes, dtype=np.uint8).reshape((Config.height, Config.width, 3))

            data = PipeData(frame=frame,
                            depth_frame=None,
                            unfiltered_frame=frame, creation_time=time.time())
            data.last_touched_process = self.name
            data.timing_info.start(f"Data Lifecycle {data.last_touched_process}")
            data.timing_info.start("Process Data", parent=f"Data Lifecycle {data.last_touched_process}")

            for filter in self.filter_configuration:
                filter.process(data)

            data.timing_info.stop("Process Data")
            data.timing_info.start("Transfer Data (SPF -> MM)", parent=f"Data Lifecycle {data.last_touched_process}")
            # print(f"Timing_Info SPF->MM: {data.timing_info}")
            data_as_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            memory_writer.write(data_as_bytes)

            del data

        memory_writer.close()
        print(f"Exiting {self.name}")