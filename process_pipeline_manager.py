import os
import queue
import time
from datetime import datetime

import cv2
import numpy as np
import torch.multiprocessing as mp
from filters.base_filter import BaseFilter
from behaviour_planner import BehaviourPlanner
from filters.draw_filter import DrawFilter
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from objects.types.video_info import VideoInfo


class ProcessPipelineManager:
    def __init__(self, parallel_config: list[list[BaseFilter]],
                 video_info: VideoInfo, save=False, args=None):
        self.parallel_processes = []
        self.draw_filter = DrawFilter(video_info=video_info)
        self.behaviour_planner = BehaviourPlanner()

        for config in parallel_config:
            pipe, subprocess_pipe = mp.Pipe()
            process = SequentialFilterProcess(config, subprocess_pipe)
            process.start()
            self.parallel_processes.append((process, pipe))

        if save:
            self.save_queue = mp.Queue()
            self.save_enabled = mp.Value('b', True)  # Shared boolean value for enabling saving

            # Start a separate process for saving frames
            self.save_process = mp.Process(target=self.save_frames, args=(self.save_queue, self.save_enabled, args))
            self.save_process.start()

    def finish_saving(self):
        with self.save_enabled.get_lock():
            if not self.save_enabled.value:
                print("Saving was not enabled")
                return

        with self.save_enabled.get_lock():
            self.save_enabled.value = False

        self.save_process.join()  # Wait for the save process to finish current frame



    @staticmethod
    def save_frames(save_queue, save_enabled, args):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"\nSaving video to: {os.path.join(args.videos_dir, f'recording_{current_time}.mp4')}")
        print(f"Video resolution: {args.width}x{args.height}")
        print(f"Video FPS: {args.fps}")
        output_video_writer = cv2.VideoWriter(
            os.path.join(args.videos_dir, f"recording_{current_time}.mp4"),
            fourcc, args.fps,
            (args.width, args.height)
        )

        while True:
            try:
                with save_enabled.get_lock():
                    if not save_enabled.value:
                        print("Saving process is stopping")
                        break
                frame = save_queue.get(block=False)
                if isinstance(frame, np.ndarray) and frame.flags.writeable:
                    output_video_writer.write(frame)
                else:
                    print(f"Invalid frame type for saving: {type(frame)}")
            except queue.Empty:
                pass

        # Get all remaining frames from the queue and save them
        while not save_queue.empty():
            print(f"Saving remaining frames, frames left:{save_queue.qsize()}")
            frame = save_queue.get()
            if frame is None:
                print("save process received None")
                continue
            output_video_writer.write(frame)

        output_video_writer.release()

    def run(self, frame, visualize: False, depth_frame=None):
        start_time = time.time()
        data: PipeData = PipeData(frame=frame, depth_frame=depth_frame, unfiltered_frame=frame.copy())
        for process, pipe in self.parallel_processes:
            pipe.send(data)

        if self.save_enabled.value:
            self.save_queue.put(data.unfiltered_frame)

        for process, pipe in self.parallel_processes:
            new_data = pipe.recv()
            data = data.merge(new_data)

        end_time = time.time()

        print(f"Parallel execution time: {end_time - start_time} seconds")

        data.command = self.behaviour_planner.run_iteration(traffic_signs=data.traffic_signs,
                                                            traffic_lights=data.traffic_lights,
                                                            pedestrians=data.pedestrians,
                                                            horizontal_lines=data.horizontal_lines)
        print(data.command)
        if visualize:
            data = self.draw_filter.process(data)

        return data