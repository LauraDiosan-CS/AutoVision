class PipeData:
    def __init__(self, frame, road_markings, heading_error, unfiltered_frame, traffic_signs, processed_frames=None):
        self.frame = frame
        self.road_markings = road_markings
        self.heading_error = heading_error
        self.unfiltered_frame = unfiltered_frame
        self.traffic_signs = traffic_signs

        if processed_frames is None:
            processed_frames = []
        self.processed_frames = processed_frames

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

        # Merge processed frames
        self.processed_frames.extend(other.processed_frames)

        return self