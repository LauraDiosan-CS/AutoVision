import torch.multiprocessing as mp

from filters.base_filter import BaseFilter
from filters.draw_filter import DrawFilter
from helpers.helpers import Timer
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from objects.types.video_info import VideoInfo


class ProcessPipelineManager:
    def __init__(self, parallel_config: list[list[BaseFilter]],
                 video_info: VideoInfo):

        self.parallel_processes = []
        self.draw_filter = DrawFilter(video_info=video_info)

        for config in parallel_config:
            pipe, subprocess_pipe = mp.Pipe()
            process = SequentialFilterProcess(config, subprocess_pipe)
            process.start()
            self.parallel_processes.append((process, pipe))

    def process_frame(self, data: PipeData, apply_draw_filter=False):
        with Timer("Parallel pipeline execution"):
            for process, pipe in self.parallel_processes:
                pipe.send(data)

            for process, pipe in self.parallel_processes:
                new_data = pipe.recv()
                data = data.merge(new_data)

        if apply_draw_filter:
            with Timer("Draw Filter"):
                data = self.draw_filter.process(data)

        return data