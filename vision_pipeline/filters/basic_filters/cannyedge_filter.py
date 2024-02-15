from copy import deepcopy

from objects.pipe_data import PipeData
from objects.video_info import VideoInfo
from vision_pipeline.filters.base_filter import BaseFilter
import cv2


class CannyEdgeFilter(BaseFilter):
    # params: low - 150, high - 250
    def __init__(self, video_info: VideoInfo, low_threshold=150, high_threshold=250):
        super().__init__(video_info=video_info)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, data: PipeData) -> PipeData:
        data.frame = cv2.Canny(data.frame, self.low_threshold, self.high_threshold)
        data.processed_frames.append(deepcopy(data.frame))
        return data