from vision_pipeline.filters.base_filter import BaseFilter
from video_info import VideoInfo
import cv2

class GrayScaleFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo):
        super().__init__(video_info=video_info, return_type="img")

    def process(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray_frame