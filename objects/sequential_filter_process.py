import torch.multiprocessing as mp

from filters.base_filter import BaseFilter


class SequentialFilterProcess(mp.Process):
    def __init__(self, filter_configuration: list[BaseFilter], in_queue: mp.Queue, out_queue: mp.Queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.filter_configuration = filter_configuration

    def run(self):
        while True:
            data = self.in_queue.get()
            for filter in self.filter_configuration:
                data = filter.process(data)
            self.out_queue.put(data)