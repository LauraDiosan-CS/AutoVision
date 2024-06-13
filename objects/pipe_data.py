from objects.types.road_info import RoadObject

class PipeData:
    def __init__(self, frame, depth_frame, unfiltered_frame):
        self.frame = frame
        self.depth_frame = depth_frame
        self.road_markings = None
        self.heading_error = None
        self.lateral_offset = None
        self.unfiltered_frame = unfiltered_frame
        self.traffic_signs: list[RoadObject] = []
        self.traffic_lights: list[RoadObject] = []
        self.pedestrians: list[RoadObject] = []
        self.horizontal_lines: list[RoadObject] = []
        self.command: str = ""
        self.processed_frames = []
        self.pipeline_execution_time = 0

    def __str__(self):
        return (f"PipeData(..., heading_error={self.heading_error}, "
                f"processed_frames_count={len(self.processed_frames)})")

    __repr__ = __str__

    def merge(self, other):
        """
        Merge data from another PipeData instance into this one.
        """
        # Check if both instances have the same field
        for field in ['road_markings', 'heading_error']:
            if getattr(self, field) is not None and getattr(other, field) is not None:
                raise ValueError(f"Both instances have a value for the field '{field}'. Cannot merge.")

        if other.road_markings is not None:
            self.road_markings = other.road_markings
        if other.heading_error is not None:
            self.heading_error = other.heading_error
        if other.lateral_offset is not None:
            self.lateral_offset = other.lateral_offset
        if other.command != "":
            self.command = other.command
        self.traffic_signs.extend(other.traffic_signs)
        self.traffic_lights.extend(other.traffic_lights)
        self.pedestrians.extend(other.pedestrians)
        self.horizontal_lines.extend(other.horizontal_lines)

        self.processed_frames.extend(other.processed_frames)

        return self