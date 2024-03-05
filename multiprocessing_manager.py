import os
import time
from datetime import datetime
import httpx
import torch.multiprocessing as mp

from behaviour_planner import BehaviourPlanner
from config import Config
from filters.base_filter import BaseFilter
from helpers.helpers import save_frames
from objects.pipe_data import PipeData
from objects.types.save_info import SaveInfo
from objects.types.video_info import VideoInfo
from process_pipeline_manager import ProcessPipelineManager


class MultiProcessingManager:
    def __init__(self, parallel_config: list[list[BaseFilter]],
                 video_info: VideoInfo, save_input=False, save_output=False):
        self.behaviour_planner = BehaviourPlanner()
        self.save_input = save_input
        self.save_output = save_output
        self.http_connection_failed_count = 0
        self.save_queue = None
        self.save_process = None
        self.save_enabled = None

        self.process_pipeline_manager = ProcessPipelineManager(
            parallel_config=parallel_config,
            video_info=video_info
        )

        if save_input or save_output:
            self.initialize_saving()

    def initialize_saving(self):
        # Start a separate process for saving frames

        self.save_queue = mp.Queue()
        self.save_enabled = mp.Value('b', True)  # Shared boolean value for enabling saving

        # Extract the name without the extension
        video_name = os.path.splitext(Config.video_name)[0]

        save_info = SaveInfo(
            video_path=os.path.join(Config.recordings_dir,
                                    f"{video_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"),
            width=Config.width,
            height=Config.height,
            fps=Config.fps
        )
        self.save_process = mp.Process(target=save_frames,
                                       args=(self.save_queue,
                                             self.save_enabled,
                                             save_info))
        self.save_process.start()

    def process_frame(self, frame, depth_frame=None, apply_draw_filter=False):
        data: PipeData = PipeData(frame=frame,
                                  depth_frame=depth_frame,
                                  unfiltered_frame=frame.copy())

        data = self.process_pipeline_manager.process_frame(
            data=data,
            apply_draw_filter=apply_draw_filter or self.save_output  # Apply draw filter if saving is enabled
        )

        # Perform behavior planning based on processed data
        data.command = self.behaviour_planner.run_iteration(
            traffic_signs=data.traffic_signs,
            traffic_lights=data.traffic_lights,
            pedestrians=data.pedestrians,
            horizontal_lines=data.horizontal_lines
        )

        self.handle_http_communication(data)

        if self.save_input and self.save_enabled is not None and self.save_enabled.value:
            self.save_queue.put(data.unfiltered_frame)
        elif self.save_output and self.save_enabled is not None and self.save_enabled.value:
            self.save_queue.put(data.frame)

        return data

    def handle_http_communication(self, data):
        if Config.command_url and self.http_connection_failed_count < 3:
            try:
                json_data = {"action": data.command.value,
                             "heading_error_degrees": data.heading_error,
                             "observed_acceleration": 0}
                start_time = time.time()
                r = httpx.post(Config.command_url, json=json_data)
                end_time = time.time()
                print(f"Httpx success execution time: {end_time - start_time} seconds")
                if r.status_code == 422:
                    print(f"Error sending command to the car: {r.text}")
            except Exception as e:
                print(f"Error connecting to the car: {e}")
                self.http_connection_failed_count += 1

    def finish_saving(self):
        with self.save_enabled.get_lock():
            if not self.save_enabled.value:
                print("Saving was not enabled")
                return

        with self.save_enabled.get_lock():
            self.save_enabled.value = False

        self.save_process.join()  # Wait for the save process to finish current frame