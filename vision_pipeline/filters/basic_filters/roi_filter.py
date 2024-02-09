import cv2
import numpy as np
from video_info import VideoInfo
from vision_pipeline.filters.base_filter import BaseFilter

class ROIFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo ):
        super().__init__(video_info=video_info, return_type="img")

    @staticmethod
    def define_roi(image, roi_bbox):
        poly= np.array([roi_bbox])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, poly, (255, 255, 255))

        return cv2.bitwise_and(image, mask), mask, poly

    def process(self, frame):
        print(self.video_roi_bbox)
        frame, _, _ = self.define_roi(frame, self.video_roi_bbox)

        return frame
