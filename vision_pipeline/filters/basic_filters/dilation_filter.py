from vision_pipeline.filters.base_filter import BaseFilter
from video_info import VideoInfo
import cv2
from numpy import uint8, ones

class DilationFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, kernel_size=5):
        super().__init__(video_info=video_info, return_type="img")
        self.kernel_size = kernel_size

    def process(self, frame):
        kernel = ones((self.kernel_size, self.kernel_size), uint8)
        dilated_frame = cv2.dilate(frame, kernel, iterations=1)
        return dilated_frame