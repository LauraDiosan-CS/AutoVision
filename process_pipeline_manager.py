import time

import torch.multiprocessing as mp

from filters.base_filter import BaseFilter
from filters.draw_filter import DrawFilter
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from objects.types.video_info import VideoInfo


class ProcessPipelineManager:
    def __init__(self, parallel_config: list[list[BaseFilter]], video_info: VideoInfo):
        self.parallel_processes = []

        self.draw_filter = DrawFilter(video_info=video_info)

        for config in parallel_config:
            pipe, subprocess_pipe = mp.Pipe()
            process = SequentialFilterProcess(config, subprocess_pipe)
            process.start()
            self.parallel_processes.append((process, pipe))

    def run(self, frame, depth_frame):
        start_time = time.time()
        data: PipeData = PipeData(frame=frame, depth_frame=depth_frame,unfiltered_frame=frame.copy())
        for process, pipe in self.parallel_processes:
            pipe.send(data)

        for process, pipe in self.parallel_processes:
            new_data = pipe.recv()
            data = data.merge(new_data)

        data = self.draw_filter.process(data)
        end_time = time.time()

        print(f"Parallel execution time: {end_time - start_time} seconds")

        return data