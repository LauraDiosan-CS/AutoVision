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
        print('roi:', roi_bbox)

        image2 = deepcopy(image)
        image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for roi_coord in roi_bbox:
            print('roi coord:', roi_coord)
            cv2.circle(image2, (roi_coord[0], roi_coord[1]), radius=5, color=(0, 0, 255))
        return cv2.bitwise_and(image, mask), mask, poly, image2

    def process(self, data: PipeData) -> PipeData:
        data.frame, _, _, image = self.define_roi(data.frame, self.video_roi_bbox)
        data.processed_frames.append(deepcopy(data.frame))
        data.processed_frames.append(image)
        return data