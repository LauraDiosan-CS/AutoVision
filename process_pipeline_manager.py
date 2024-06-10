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
            process = SequentialFilterProcess(config, subprocess_pipe, name=config[-1].__class__.__name__)
            process.start()
            self.parallel_processes.append((process, pipe))

    def process_frame(self, data: PipeData, apply_draw_filter=False):
        timer.start('Apply All Filters in Parallel', parent='Process Frame')

        for process, pipe in self.parallel_processes:
            timer.start(f"Process Data {process.name}", parent="Apply All Filters in Parallel")
            timer.start('Send Data', parent=f'Process Data {process.name}')
            pipe.send(data)
            timer.stop('Send Data')
        for process, pipe in self.parallel_processes:
            timer.start('Recv Data', parent=f'Process Data {process.name}')
            new_data = pipe.recv()
            timer.stop('Recv Data')
            data = data.merge(new_data)
            timer.stop(f'Process Data {process.name}')
        timer.stop('Apply All Filters in Parallel')

        # Perform behavior planning based on processed data
        data.command = self.behaviour_planner.run_iteration(
            traffic_signs=data.traffic_signs,
            traffic_lights=data.traffic_lights,
            pedestrians=data.pedestrians,
            horizontal_lines=data.horizontal_lines
        )

        timer.start('Draw onto Frame', parent='Process Frame')
        if apply_draw_filter:
            data = self.draw_filter.process(data)
        timer.stop('Draw onto Frame')
        return data