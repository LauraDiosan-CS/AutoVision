import pickle
import time
import numpy as np

import multiprocessing as mp

from ripc import SharedMemoryReader, SharedMemoryWriter

from configuration.config import Config
from perception.filters.base_filter import BaseFilter
from perception.objects.pipe_data import PipeData
from processes.controlled_process import ControlledProcess

class SequentialFilterProcess(ControlledProcess):
    __slots__ = ['filters', 'keep_running', 'last_processed_frame_version', 'artificial_delay']

    def __init__(self, filters: list[BaseFilter], keep_running: mp.Value, last_processed_frame_version: mp.Value, artificial_delay: float = 0.0, program_start_time: float = 0.0, process_name: str = None):
        super().__init__(name=process_name, program_start_time=program_start_time)
        self.filters = filters
        self.keep_running = keep_running
        self.last_processed_frame_version = last_processed_frame_version
        self.artificial_delay = artificial_delay

    def run(self):
        try:
            memory_writer = SharedMemoryWriter(name=self.name, size=Config.max_pipe_data_size)
            self.finish_setup()
            video_feed = SharedMemoryReader(name=Config.video_feed_memory_name)

            while self.keep_running.value:
                frame_as_bytes = video_feed.blocking_read()

                if frame_as_bytes is None: # End of video
                    break

                self.last_processed_frame_version.value = video_feed.last_read_version()

                frame = np.frombuffer(frame_as_bytes, dtype=np.uint8).reshape((Config.height, Config.width, 3))
                data = PipeData(frame=frame,
                                frame_version=video_feed.last_read_version(),
                                depth_frame=None, # currently only available in real-time mode
                                raw_frame=frame,
                                creation_time=time.perf_counter(),
                                last_filter_process_name=self.name)

                dl = f"Data Lifecycle {self.name[0]}"
                pd = f"Process Data {self.name[0]}"
                tf = f"Transfer Data {self.name[0]}"
                data.timing_info.start(dl)
                data.timing_info.start(pd, parent=dl)

                if self.artificial_delay > 0:
                    time.sleep(self.artificial_delay)

                for filter in self.filters:
                    filter.process(data)

                data.timing_info.stop(pd)
                data.timing_info.start(tf, parent=dl)

                data_as_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                memory_writer.write(data_as_bytes)

                del data

            memory_writer.close()
        except Exception as e:
            print(f"[{self.name}] Error: {e}")