import time

import torch.multiprocessing as mp

from filters.base_filter import BaseFilter
from objects.pipe_data import PipeData


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
                data: PipeData = filter.process(data)

            end_time = time.time()
            data.pipeline_execution_time = end_time - start_time

            self.pipe.send(data)