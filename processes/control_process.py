import json
import pickle
import time

import torch.multiprocessing as mp
import urllib3
from rs_ipc import SharedMessage, OperationMode

from configuration.config import Config
from control.pid_controller import PIDController
from perception.objects.pipe_data import PipeData
from planning.behaviour_planner import BehaviourPlanner


class Control(mp.Process):
    def __init__(self, keep_running: mp.Value):
        super().__init__()
        self.steering_pid = None
        self.keep_running = keep_running

    def run(self):
        try:
            behaviour_planner = BehaviourPlanner()
            self.steering_pid = PIDController(kp=0.5, ki=0.0, kd=0.1)
            memory_reader: SharedMessage = SharedMessage.open(
                Config.control_loop_memory_name, OperationMode.ReadSync()
            )

            while self.keep_running:
                pipe_data_bytes = memory_reader.read(block=True)
                if pipe_data_bytes is None:
                    break

                pipe_data: PipeData = pickle.loads(pipe_data_bytes)

                # Perform behavior planning based on processed data
                behaviour = behaviour_planner.run_iteration(
                    traffic_signs=pipe_data.traffic_signs,
                    traffic_lights=pipe_data.traffic_lights,
                    pedestrians=pipe_data.pedestrians,
                    horizontal_lines=pipe_data.horizontal_lines,
                )

                normalized_steering_angle = self.compute_normalized_steering_angle(
                    pipe_data.heading_error_degrees, pipe_data.lateral_offset
                )

                json_data = {
                    "normalized_steering_angle": normalized_steering_angle,
                    "longitudinal_velocity": 1.0,
                }

                print(f"[Controller] Sending control data: {json_data}")
        except Exception as e:
            print(f"[Controller] Error: {e}")
            self.keep_running.value = False

    def compute_normalized_steering_angle(self, heading_error, lateral_offset):
        if heading_error is None or lateral_offset is None:
            return 0.0

        MAX_HEADING_ANGLE = 90
        HEADING_ERROR_WEIGHT = 0.75
        LATERAL_ERROR_WEIGHT = 1.35

        # clamp the lateral offset to the range [-1, 1]
        normalized_lateral_offset = max(-1.0, min(1.0, lateral_offset))

        normalized_heading_error = heading_error / MAX_HEADING_ANGLE

        # Correct the heading error based on the lateral offset
        # positive value means car is on the right side of the road
        # corrected_heading_error = w1 * heading_error + w2 * lateral_offset
        corrected_normalized_heading_error = (
            HEADING_ERROR_WEIGHT * normalized_heading_error
            + LATERAL_ERROR_WEIGHT * normalized_lateral_offset
        )

        # Increase/Decrease Normalized Steering Angle in proportion to the Normalized Heading Error
        normalized_steering_angle = self.steering_pid.compute(
            corrected_normalized_heading_error
        )

        return normalized_steering_angle