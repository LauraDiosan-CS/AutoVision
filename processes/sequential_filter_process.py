import pickle
import time
import numpy as np

import multiprocessing as mp

from ripc import SharedMessage, OpenMode

from configuration.config import Config
from perception.filters.base_filter import BaseFilter
from perception.objects.pipe_data import PipeData
from processes.controlled_process import ControlledProcess


class SequentialFilterProcess(ControlledProcess):
    __slots__ = ['filters', 'keep_running', 'last_processed_frame_version', 'artificial_delay']

    def __init__(self, filters: list[BaseFilter], keep_running: mp.Value, debug_pipe: mp.Pipe,
                 last_processed_frame_version: mp.Value, artificial_delay: float = 0.0,
                 program_start_time: float = 0.0, process_name: str = None):
        super().__init__(name=process_name, program_start_time=program_start_time)
        self.filters = filters
        self.keep_running = keep_running
        self.last_processed_frame_version = last_processed_frame_version
        self.debug_pipe = debug_pipe
        self.artificial_delay = artificial_delay

    def run(self):
        try:
            pipeline_shm = SharedMessage.open(Config.shm_base_name + self.name, mode=OpenMode.WriteOnly)
            video_feed_shm = SharedMessage.open(Config.video_feed_memory_name, mode=OpenMode.ReadOnly)

            processed_frame_indexes = []

            dl = f"Data Lifecycle {self.name[0]}"
            pd = f"Process Data {self.name[0]}"
            tf = f"Transfer Data {self.name[0]}"
            while self.keep_running.value:
                frame_as_bytes = video_feed_shm.blocking_read()

                if frame_as_bytes is None: # End of video
                    break

                self.last_processed_frame_version.value = video_feed_shm.last_read_version()
                processed_frame_indexes.append(self.last_processed_frame_version.value)

                frame = np.frombuffer(frame_as_bytes, dtype=np.uint8).reshape((Config.height, Config.width, 3))
                data = PipeData(frame=frame,
                                frame_version=video_feed_shm.last_read_version(),
                                depth_frame=None, # currently only available in real-time mode
                                raw_frame=frame,
                                creation_time=time.perf_counter(),
                                last_pipeline_name=self.name)

                data.timing_info.start(dl)
                data.timing_info.start(pd, parent=dl)

                if self.artificial_delay > 0:
                    time.sleep(self.artificial_delay)

                for filter in self.filters:
                    filter.process(data)

                data.timing_info.stop(pd)
                data.timing_info.start(tf, parent=dl)

                data_as_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                pipeline_shm.write(data_as_bytes)

                del data

            pipeline_shm.close()

            self.debug_pipe.send(processed_frame_indexes)
            self.debug_pipe.close()
        except Exception as e:
            print(f"[{self.name}] Error: {e}")