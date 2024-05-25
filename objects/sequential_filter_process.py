import pickle
import time

import multiprocessing as mp

from filters.base_filter import BaseFilter
from helpers.shared_memory import SharedMemoryReader, SharedMemoryWriter
from config import Config

class SequentialFilterProcess(mp.Process):
    __slots__ = ['filter_configuration', 'keep_running']

    def __init__(self, filter_configuration: list[BaseFilter], process_name: str, keep_running: mp.Value):
        super().__init__(name=process_name)
        self.filter_configuration = filter_configuration
        self.keep_running = keep_running

    def run(self):
        memory_reader = SharedMemoryReader(topic=Config.video_feed_shared_memory_name, create=False)
        memory_writer = SharedMemoryWriter(topic=self.name, create=False)

        while self.keep_running:
            frame_as_bytes = memory_reader.read()
            while frame_as_bytes is None:
                if not self.keep_running.value:
                    print(f"Exiting {self.name}")
                    return
                time.sleep(0.01)
                frame_as_bytes = memory_reader.read()

            frame = np.frombuffer(frame_as_bytes, dtype=np.uint8).reshape((Config.height, Config.width, 3))

            data = PipeData(frame=frame,
                            depth_frame=None,
                            unfiltered_frame=frame.copy())

            # last_data.last_touched_process = self.name
            for filter in self.filter_configuration:
                data = filter.process(data)

            data_as_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            memory_writer.write(data_as_bytes)

        print(f"Exiting {self.name}")


# class SequentialFilterProcess(mp.Process):
#     def __init__(self, filter_configuration: list[BaseFilter], in_queue: mp.Queue, out_queue: mp.Queue):
#         super().__init__()
#         self.in_queue = in_queue
#         self.out_queue = out_queue
#         self.filter_configuration = filter_configuration
#
#     def run(self):
#         while True:
#             data = self.in_queue.get()
#             for filter in self.filter_configuration:
#                 data = filter.process(data)
#             self.out_queue.put(data)