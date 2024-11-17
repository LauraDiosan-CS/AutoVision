from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from perception.objects.road_info import RoadMarkings, RoadObject
from perception.objects.timing_info import TimingInfo


@dataclass(slots=True)
class PipeData:
    frame: np.array
    frame_version: int
    depth_frame: np.array
    unfiltered_frame: np.array
    creation_time: float
    last_touched_process: str
    timing_info: TimingInfo = field(default_factory=TimingInfo)
    processed_frames: dict[str, list[np.array]] = field(default_factory=dict)
    road_markings: Optional[RoadMarkings] = None
    heading_error: Optional[float] = None
    lateral_offset: Optional[float] = None
    traffic_signs: list[RoadObject] = field(default_factory=list)
    traffic_lights: list[RoadObject] = field(default_factory=list)
    pedestrians: list[RoadObject] = field(default_factory=list)
    horizontal_lines: list[RoadObject] = field(default_factory=list)
    behaviour: str = None


    def add_processed_frame(self, frame):
        """
        Add a processed frame to the data.
        """
        if self.last_touched_process not in self.processed_frames:
            self.processed_frames[self.last_touched_process] = [frame]
        else:
            self.processed_frames[self.last_touched_process].append(frame)


    def merge(self, new_pipe_data: 'PipeData') -> 'PipeData':
        """
        Merge data from another PipeData instance into this one.
        """
        self.timing_info.append_hierarchy(new_pipe_data.timing_info)

        if new_pipe_data.frame_version > self.frame_version:
            self.frame = new_pipe_data.frame
            self.depth_frame = new_pipe_data.depth_frame
            self.frame_version = new_pipe_data.frame_version
            self.unfiltered_frame = new_pipe_data.unfiltered_frame

        self.last_touched_process = new_pipe_data.last_touched_process

        for process_name, frames in new_pipe_data.processed_frames.items():
            self.processed_frames[process_name] = frames

        if new_pipe_data.road_markings is not None:
            self.road_markings = new_pipe_data.road_markings
        if new_pipe_data.heading_error is not None:
            self.heading_error = new_pipe_data.heading_error
        if new_pipe_data.lateral_offset is not None:
            self.lateral_offset = new_pipe_data.lateral_offset
        if new_pipe_data.behaviour is not None:
            self.behaviour = new_pipe_data.behaviour
        if new_pipe_data.traffic_signs is not None: # [] is a valid value
            self.traffic_signs = new_pipe_data.traffic_signs
        if new_pipe_data.traffic_lights is not None:
            self.traffic_lights = new_pipe_data.traffic_lights
        if new_pipe_data.pedestrians is not None:
            self.pedestrians = new_pipe_data.pedestrians
        if new_pipe_data.horizontal_lines is not None:
            self.horizontal_lines = new_pipe_data.horizontal_lines


        # attributes = [
        #     'frame',
        #     'depth_frame',
        #     'unfiltered_frame',
        #     'road_markings',
        #     'heading_error',
        #     'last_touched_process',
        #     'lateral_offset',
        #     'command',
        #     'traffic_signs',
        #     'traffic_lights',
        #     'pedestrians',
        #     'horizontal_lines'
        # ]
        #
        # for attr in attributes:
        #     new_value = getattr(new_pipe_data, attr)
        #     print(f"Attribute: {attr}, new_value: {new_value}")
        #     # Verify if the new pipe data has a valid value for the attribute
        #     if new_value or new_value == 0:  # Check for non-None or valid 0
        #         setattr(self, attr, new_value)

        return self