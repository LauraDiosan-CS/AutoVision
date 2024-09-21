import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
from mpmath import timing

from helpers.timing_info import TimingInfo
from objects.types.road_info import RoadObject, RoadMarkings


@dataclass(slots=True)
class PipeData:
    frame: np.array
    depth_frame: np.array
    unfiltered_frame: np.array
    creation_time: float
    timing_info: TimingInfo = field(default_factory=TimingInfo)
    processed_frames: Dict[str, List[np.array]] = field(default_factory=dict)
    last_touched_process: str = ""
    road_markings: Optional[RoadMarkings] = None
    heading_error: Optional[float] = None
    lateral_offset: Optional[float] = None
    traffic_signs: List[RoadObject] = field(default_factory=list)
    traffic_lights: List[RoadObject] = field(default_factory=list)
    pedestrians: List[RoadObject] = field(default_factory=list)
    horizontal_lines: List[RoadObject] = field(default_factory=list)
    command: str = ""


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
        print(f"Merge timings: {self.timing_info.hierarchy}")

        attributes = [
            'frame',
            'depth_frame',
            'unfiltered_frame',
            'road_markings',
            'heading_error',
            'last_touched_process',
            'lateral_offset',
            'command',
            'traffic_signs',
            'traffic_lights',
            'pedestrians',
            'horizontal_lines'
        ]

        for attr in attributes:
            new_value = getattr(new_pipe_data, attr)

            # Verify if the new pipe data has a valid value for the attribute
            if new_value or new_value == 0:  # Check for non-None or valid 0
                setattr(self, attr, new_value)

        for process_name, frames in new_pipe_data.processed_frames.items():
            self.processed_frames[process_name] = frames

        return self

#         if new_pipe_data.frame is not None:
#             self.frame = new_pipe_data.frame
#         if new_pipe_data.depth_frame is not None:
#             self.depth_frame = new_pipe_data.depth_frame
#         if new_pipe_data.unfiltered_frame is not None:
#             self.unfiltered_frame = new_pipe_data.unfiltered_frame
#         if new_pipe_data.road_markings is not None:
#             self.road_markings = new_pipe_data.road_markings
#         if new_pipe_data.heading_error is not None:
#             self.heading_error = new_pipe_data.heading_error
#         if new_pipe_data.last_touched_process != "":
#             self.last_touched_process = new_pipe_data.last_touched_process
#         if new_pipe_data.lateral_offset is not None:
#             self.lateral_offset = new_pipe_data.lateral_offset
#         if new_pipe_data.command != "":
#             self.command = new_pipe_data.command
#         if new_pipe_data.traffic_signs:
#             self.traffic_signs = new_pipe_data.traffic_signs
#         if new_pipe_data.traffic_lights:
#             self.traffic_lights = new_pipe_data.traffic_lights
#         if new_pipe_data.pedestrians:
#             self.pedestrians = new_pipe_data.pedestrians
#         if new_pipe_data.horizontal_lines:
#             self.horizontal_lines = new_pipe_data.horizontal_lines