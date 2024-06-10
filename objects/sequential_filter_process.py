import time

import torch.multiprocessing as mp

from filters.base_filter import BaseFilter


class SequentialFilterProcess(mp.Process):
    def __init__(self, filter_configuration: list[BaseFilter], pipe: mp.Pipe, name=None):
        super().__init__(name=name)
        self.pipe = pipe
        self.filter_configuration = filter_configuration

    def run(self):
        while True:
            data = self.pipe.recv()
            start_time = time.time()
            for filter in self.filter_configuration:
                data = filter.process(data)
            end_time = time.time()
            print(f"{self.name} took {(end_time - start_time) * 1000} ms")

            self.pipe.send(data)