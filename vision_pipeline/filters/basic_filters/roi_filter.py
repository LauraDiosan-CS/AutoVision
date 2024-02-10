from copy import deepcopy

import cv2
import numpy as np
from objects.pipe_data import PipeData
from objects.video_info import VideoInfo
from vision_pipeline.filters.base_filter import BaseFilter


class ROIFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo):
        super().__init__(video_info=video_info)

    @staticmethod
    def define_roi(image, roi_bbox):
        poly = np.array([roi_bbox])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, poly, (255, 255, 255))

        return cv2.bitwise_and(image, mask), mask, poly

    def process(self, data: PipeData) -> PipeData:
        data.frame, _, _ = self.define_roi(data.frame, self.video_roi_bbox)
        data.processed_frames.append(deepcopy(data.frame))
        return data