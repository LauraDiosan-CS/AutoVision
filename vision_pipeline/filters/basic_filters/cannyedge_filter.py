from vision_pipeline.filters.base_filter import BaseFilter
from video_info import VideoInfo
import cv2

class CannyEdgeFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, low_threshold=150, high_threshold=250):
        super().__init__(video_info=video_info, return_type="img")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, frame):
        edges_frame = cv2.Canny(frame, self.low_threshold, self.high_threshold)
        return edges_frame