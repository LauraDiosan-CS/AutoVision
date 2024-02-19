import torch.multiprocessing as mp

from filters.base_filter import BaseFilter


class SequentialFilterProcess(mp.Process):
    def __init__(self, filter_configuration: list[BaseFilter], pipe: mp.Pipe):
        super().__init__()
        self.pipe = pipe
        self.filter_configuration = filter_configuration

    def run(self):
        while True:
            data = self.pipe.recv()

            for filter in self.filter_configuration:
                data = filter.process(data)

            self.pipe.send(data)