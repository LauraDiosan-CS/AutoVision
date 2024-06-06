import json
import pickle
import time

import torch.multiprocessing as mp
import urllib3
from ripc import SharedMemoryReader

from config import Config
from controllers.behaviour_planner import BehaviourPlanner
from objects.pipe_data import PipeData


class Controller(mp.Process):
    def __init__(self, keep_running: mp.Value):
        super().__init__()
        self.keep_running = keep_running


    def run(self):
        behaviour_planner = BehaviourPlanner()
        http_pool = urllib3.PoolManager()
        memory_reader = SharedMemoryReader(name=Config.composite_pipe_memory_name)

        while self.keep_running:
            pipe_data_bytes = memory_reader.read()
            while pipe_data_bytes is None:
                if not self.keep_running.value:
                    print(f"Exiting {self.name}")
                    return
                time.sleep(0.01)
                pipe_data_bytes = memory_reader.read()

            pipe_data: PipeData = pickle.loads(pipe_data_bytes)

            # Perform behavior planning based on processed data
            pipe_data.command = behaviour_planner.run_iteration(
                    traffic_signs=pipe_data.traffic_signs,
                    traffic_lights=pipe_data.traffic_lights,
                    pedestrians=pipe_data.pedestrians,
                    horizontal_lines=pipe_data.horizontal_lines
                )

            self.handle_http_communication(pipe_data, http_pool)

    def handle_http_communication(self, data, http_pool):
        if Config.command_url and Config.http_connection_failed_limit < 3:
            try:
                json_data = {"action": data.command.value,
                             "heading_error_degrees": data.heading_error,
                             "observed_acceleration": 0}
                r = http_pool.request('POST', Config.command_url, headers={'Content-Type': 'application/json'},
                                           body=json.dumps(json_data))
            except Exception as e:
                print(f"Error connecting to the car: {e}")