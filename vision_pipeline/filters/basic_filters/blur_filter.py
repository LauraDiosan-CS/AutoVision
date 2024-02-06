from vision_pipeline.filters.base_filter import BaseFilter
from video_info import VideoInfo
import cv2

class BlurFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, kernel_size=5):
        super().__init__(video_info=video_info, return_type="img")
        self.kernel_size = kernel_size

    def process(self, frame):
        blurred_frame = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), 0)
        return blurred_frame