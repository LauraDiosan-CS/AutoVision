from copy import deepcopy

from vision_pipeline.lane_detection import LaneDetect
from vision_pipeline.traffic_objects_detection import SignsDetect
from fastcore.parallel import parallel
import numpy as np


class Pipeline:
    def __init__(self, args):
        self.signs_model = args.yolo_signs
        self.lane_detection = LaneDetect()
        self.sign_detection = SignsDetect(self.signs_model)
        self.filters = [self.apply_sign_detection, self.apply_lane_detection]

    def apply_lane_detection(self, frame):
        return self.lane_detection.process(frame)

    def apply_sign_detection(self, frame):
        return self.sign_detection.process(frame)

    # sequential processing of the pipeline
    def run_seq(self, frame):
        processed_frames = []
        for filt in self.filters:
            frame = filt(frame)
            processed_frames.append(deepcopy(frame))

        return processed_frames

    # parallel processing of the pipeline
    def run_par(self, frame):
        # it's very slow for some reason =)) can not figure it out
        processing_pairs = [(frame, filter) for filter in self.filters]
        processed_frames = parallel(self.apply_filter, processing_pairs, n_workers=2)

        combined_frame = np.maximum(*processed_frames)
        processed_frames.append(combined_frame)

        return processed_frames

    def apply_filter(self, processing_pairs):
        frame, filter = processing_pairs
        return filter(frame)