class PipeData:
    def __init__(self, frame, road_markings, steering_angle, unfiltered_frame, processed_frames=None):
        self.frame = frame
        self.road_markings = road_markings
        self.steering_angle = steering_angle
        self.unfiltered_frame = unfiltered_frame

        if processed_frames is None:
            processed_frames = []
        self.processed_frames = processed_frames