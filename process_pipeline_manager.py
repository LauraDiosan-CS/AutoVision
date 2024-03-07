import time
import torch.multiprocessing as mp

from filters.base_filter import BaseFilter
from filters.draw_filter import DrawFilter
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
        start_time = time.time()

        for process, pipe in self.parallel_processes:
            pipe.send(data)

        for process, pipe in self.parallel_processes:
            new_data = pipe.recv()
            data = data.merge(new_data)

        end_time = time.time()

        print(f"Parallel execution time: {end_time - start_time} seconds")


        if apply_draw_filter:
            data = self.draw_filter.process(data)

        return data