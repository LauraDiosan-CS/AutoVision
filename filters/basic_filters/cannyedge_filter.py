from objects.pipe_data import PipeData
from filters.base_filter import BaseFilter
import cv2

from objects.types.video_info import VideoInfo


class CannyEdgeFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, low_threshold, high_threshold):
        super().__init__(video_info=video_info, visualize=visualize)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, data: PipeData) -> PipeData:
        data.frame = cv2.Canny(data.frame, self.low_threshold, self.high_threshold)

        return super().process(data)