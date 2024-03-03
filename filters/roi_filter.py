import cv2
import numpy as np
from objects.pipe_data import PipeData
from objects.types.video_info import VideoInfo
from filters.base_filter import BaseFilter


class ROIFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, roi_type: str):
        super().__init__(video_info=video_info, visualize=visualize)

        if not isinstance(roi_type, str) or roi_type not in self.video_rois:
            raise ValueError(f"Invalid roi_type {roi_type}")

        self.roi_bounding_box = self.video_rois.get(roi_type)

    @staticmethod
    def define_roi(image, roi_bbox):
        poly = np.array([roi_bbox])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, poly, (255, 255, 255))

        return cv2.bitwise_and(image, mask), mask, poly

    def process(self, data: PipeData) -> PipeData:
        data.frame, _, _ = self.define_roi(data.frame, self.roi_bounding_box)

        return super().process(data)