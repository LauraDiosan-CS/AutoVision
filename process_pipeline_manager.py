import torch.multiprocessing as mp

from behaviour_planner import BehaviourPlanner
from filters.base_filter import BaseFilter
from filters.draw_filter import DrawFilter
from helpers.timer import timer
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from objects.types.video_info import VideoInfo


class ProcessPipelineManager:
    def __init__(self, parallel_config: list[list[BaseFilter]],
                 video_info: VideoInfo):

        self.parallel_processes = []
        self.draw_filter = DrawFilter(video_info=video_info)
        self.behaviour_planner = BehaviourPlanner()


        for config in parallel_config:
            pipe, subprocess_pipe = mp.Pipe()
            process = SequentialFilterProcess(config, subprocess_pipe)
            process.start()
            self.parallel_processes.append((process, pipe))

    def process_frame(self, data: PipeData, apply_draw_filter=False):
        timer.start('process_frame_parallel')

        for process, pipe in self.parallel_processes:
            pipe.send(data)

        for process, pipe in self.parallel_processes:
            new_data = pipe.recv()
            data = data.merge(new_data)
        timer.stop('process_frame_parallel')

        timer.start('behaviour_planner')
        # Perform behavior planning based on processed data
        data.command = self.behaviour_planner.run_iteration(
            traffic_signs=data.traffic_signs,
            traffic_lights=data.traffic_lights,
            pedestrians=data.pedestrians,
            horizontal_lines=data.horizontal_lines
        )
        timer.stop('behaviour_planner')

        timer.start('draw_filter')
        if apply_draw_filter:
            data = self.draw_filter.process(data)
        timer.stop('draw_filter')

        return data