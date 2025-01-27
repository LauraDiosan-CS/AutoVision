from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from perception.objects.road_info import RoadMarkings, RoadObject
from perception.objects.timing_info import TimingInfo


@dataclass(slots=True)
class PipeData:
    frame: np.array
    frame_version: int
    depth_frame: np.array
    raw_frame: np.array

    creation_time: float
    last_pipeline_name: str

    timing_info: TimingInfo = field(default_factory=TimingInfo)
    processed_frames: dict[str, list[np.array]] = field(default_factory=dict)

    # Pipeline specific data
    road_markings: Optional[RoadMarkings] = None
    heading_error_degrees: Optional[float] = None
    lateral_offset: Optional[float] = None
    traffic_signs: list[RoadObject] = None
    traffic_lights: list[RoadObject] = None
    pedestrians: list[RoadObject] = None
    horizontal_lines: list[RoadObject] = None

    def add_processed_frame(self, frame, downscale_factor=2):
        """
        Add a processed frame to the data after downscaling its resolution.

        Parameters:
        - frame: The image/frame to be processed and added.
        - downscale_factor (int or float): The factor by which to downscale the frame.
                                           Must be greater than 1. (1 means no downscaling).
        """
        if downscale_factor <= 1:
            raise ValueError("downscale_factor must be greater than 1.")

        # Downscale the frame if the factor is not 1
        if downscale_factor > 1:
            height, width = frame.shape[:2]
            new_width = int(width / downscale_factor)
            new_height = int(height / downscale_factor)
            # Ensure new dimensions are at least 1x1
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Add the (possibly downscaled) frame to processed_frames
        self.processed_frames.setdefault(self.last_pipeline_name, []).append(frame)

    def merge(self, new_pipe_data: 'PipeData') -> 'PipeData':
        """
        Merge data from another PipeData instance into this one.
        """
        self.timing_info.append_hierarchy(new_pipe_data.timing_info)

        if new_pipe_data.frame_version > self.frame_version:
            self.frame = new_pipe_data.frame
            self.depth_frame = new_pipe_data.depth_frame
            self.frame_version = new_pipe_data.frame_version
            self.raw_frame = new_pipe_data.raw_frame

        self.last_pipeline_name = new_pipe_data.last_pipeline_name

        for process_name, frames in new_pipe_data.processed_frames.items():
            self.processed_frames[process_name] = frames

        if new_pipe_data.road_markings is not None:
            self.road_markings = new_pipe_data.road_markings
        if new_pipe_data.heading_error_degrees is not None:
            self.heading_error_degrees = new_pipe_data.heading_error_degrees
        if new_pipe_data.lateral_offset is not None:
            self.lateral_offset = new_pipe_data.lateral_offset
        if new_pipe_data.traffic_signs is not None: # [] is a valid value
            self.traffic_signs = new_pipe_data.traffic_signs
        if new_pipe_data.traffic_lights is not None:
            self.traffic_lights = new_pipe_data.traffic_lights
        if new_pipe_data.pedestrians is not None:
            self.pedestrians = new_pipe_data.pedestrians
        if new_pipe_data.horizontal_lines is not None:
            self.horizontal_lines = new_pipe_data.horizontal_lines

        return self