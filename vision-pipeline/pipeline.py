from lane_detection import LaneDetect
from traffic_objects_detection import SignsDetect
from fastcore.parallel import parallel
import numpy as np
import cv2

class Pipeline:
    def __init__(self, args):
        self.signs_model = args.yolo_signs
        self.lane_detection = LaneDetect()
        self.sign_detection = SignsDetect(self.signs_model)
        self.filters = [self.apply_sign_detection, self.appply_lane_detection]
   
    def appply_lane_detection(self, frame):
        return self.lane_detection.process(frame)
    
    def apply_sign_detection(self, frame):
        return self.sign_detection.process(frame)
    
    # sequential processing of the pipeline
    def run_seq(self, frame):
        for filter in self.filters:
            frame = filter(frame)

        return frame
    
    # parallel processing of the pipeline
    def run_par(self, frame):
        # it's very slow for some reason =)) can not figure it out
        processing_pairs = [(frame, filter) for filter in self.filters]
        processed_frames = parallel(self.apply_filter, processing_pairs, n_workers=5)
        combined_frame = np.maximum(*processed_frames)

        return combined_frame

    def apply_filter(self, processing_step):
        frame, filter = processing_step
        return filter(frame)