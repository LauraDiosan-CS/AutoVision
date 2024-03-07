import json

import torch.multiprocessing as mp
import urllib3

from behaviour_planner import BehaviourPlanner
from config import Config


class Controller(mp.Process):
    def __init__(self, pipe: mp.Pipe):
        super().__init__()
        self.pipe = pipe
        self.http_connection_failed_count = 0
        self.http_pool = urllib3.PoolManager()
        self.behaviour_planner = BehaviourPlanner()

    def run(self):
        while True:
            data = self.pipe.recv()

            # Perform behavior planning based on processed data
            data.command = self.behaviour_planner.run_iteration(
                traffic_signs=data.traffic_signs,
                traffic_lights=data.traffic_lights,
                pedestrians=data.pedestrians,
                horizontal_lines=data.horizontal_lines
            )

            self.handle_http_communication(data)

    def handle_http_communication(self, data):
        if Config.command_url and self.http_connection_failed_count < 3:
            try:
                json_data = {"action": data.command.value,
                             "heading_error_degrees": data.heading_error,
                             "observed_acceleration": 0}
                r = self.http_pool.request('POST', Config.command_url, headers={'Content-Type': 'application/json'},
                                           body=json.dumps(json_data))
            except Exception as e:
                print(f"Error connecting to the car: {e}")
                self.http_connection_failed_count += 1