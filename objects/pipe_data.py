import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from objects.types.road_info import RoadObject, RoadMarkings


@dataclass(slots=True)
class PipeData:
    frame: np.array
    depth_frame: np.array
    unfiltered_frame: np.array
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
    pipeline_execution_time: float = 0
    creation_time: float = time.time()

    def add_processed_frame(self, frame):
        """
        Add a processed frame to the data.
        """
        if self.last_touched_process not in self.processed_frames:
            self.processed_frames[self.last_touched_process] = [frame]
        else:
            self.processed_frames[self.last_touched_process].append(frame)

    def merge(self, other):
        """
        Merge data from another PipeData instance into this one.
        """
        if other.frame is not None:
            self.frame = other.frame
        if other.depth_frame is not None:
            self.depth_frame = other.depth_frame
        if other.unfiltered_frame is not None:
            self.unfiltered_frame = other.unfiltered_frame
        if other.road_markings is not None:
            self.road_markings = other.road_markings
        if other.heading_error is not None:
            self.heading_error = other.heading_error
        if other.last_touched_process != "":
            self.last_touched_process = other.last_touched_process
        if other.lateral_offset is not None:
            self.lateral_offset = other.lateral_offset
        if other.command != "":
            self.command = other.command
        if other.traffic_signs and len(other.traffic_signs) > 0:
            self.traffic_signs = other.traffic_signs
        if other.traffic_lights and len(other.traffic_lights) > 0:
            self.traffic_lights = other.traffic_lights
        if other.pedestrians and len(other.pedestrians) > 0:
            self.pedestrians = other.pedestrians
        if other.horizontal_lines and len(other.horizontal_lines) > 0:
            self.horizontal_lines = other.horizontal_lines

        for process_name, frames in other.processed_frames.items():
            self.processed_frames[process_name] = frames

        return self