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
    __slots__ = ['filters', 'keep_running', 'last_processed_frame_version', 'artificial_delay']

    def __init__(self, filters: list[BaseFilter], keep_running: mp.Value, last_processed_frame_version: mp.Value, artificial_delay: float = 0.0, process_name=None):
        super().__init__(name=process_name)
        self.filters = filters
        self.keep_running = keep_running
        self.last_processed_frame_version = last_processed_frame_version
        self.artificial_delay = artificial_delay

    def run(self):
        memory_writer = SharedMemoryWriter(name=self.name, size=Config.pipe_memory_size)
        self.finish_setup()
        video_feed = SharedMemoryReader(name=Config.video_feed_memory_name)

        while self.keep_running.value:
            frame_as_bytes = video_feed.blocking_read()

            if frame_as_bytes is None: # End of video
                break
            print(f"{self.name} received frame at {(time.perf_counter() - Config.program_start_time):.2f} s")

            self.last_processed_frame_version.value = video_feed.last_read_version()

            frame = np.frombuffer(frame_as_bytes, dtype=np.uint8).reshape((Config.height, Config.width, 3))

            data = PipeData(frame=frame,
                            depth_frame=None,
                            unfiltered_frame=frame,
                            creation_time=time.perf_counter(),
                            last_touched_process=self.name)
            data.timing_info.start(f"Data Lifecycle {data.last_touched_process}")
            data.timing_info.start("Process Data", parent=f"Data Lifecycle {data.last_touched_process}")
            time.sleep(self.artificial_delay)

            for filter in self.filters:
                filter.process(data)

            data.timing_info.stop("Process Data")
            data.timing_info.start("Transfer Data (SPF -> MM)", parent=f"Data Lifecycle {data.last_touched_process}")
            # print(f"Timing_Info SPF({self.name})->MM: {data.timing_info}")
            data_as_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            memory_writer.write(data_as_bytes)

            del data
            print(f"{self.name} finished processing frame at {(time.perf_counter() - Config.program_start_time):.2f} s")

        memory_writer.close()
        print(f"Exiting {self.name}")