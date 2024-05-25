import time

import torch.multiprocessing as mp

from filters.base_filter import BaseFilter
from helpers.shared_memory import SharedMemoryReader, SharedMemoryWriter

class SequentialFilterProcess(mp.Process):
    def __init__(self, filter_configuration: list[BaseFilter], process_name: str, keep_running: mp.Value):
        super().__init__(name=process_name)
        self.filter_configuration = filter_configuration
        self.keep_running = keep_running

        self.shared_memo_reader = SharedMemoryReader(topic=Config.video_feed_shared_memory_name)
        self.shared_memo_writer = SharedMemoryWriter(topic=process_name + "_sm", size=100 * 1024 * 1024)

    def run(self):
        while True:
            if not self.keep_running:
                break

            data = self.shared_memo_reader.read()

            while data is None:
                time.sleep(0.01)
                data = self.shared_memo_reader.read()

            # last_data.last_touched_process = self.name
            for filter in self.filter_configuration:
                data = filter.process(data)
            data_as_bytes = data
            self.shared_memo_writer.write(data_as_bytes)

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