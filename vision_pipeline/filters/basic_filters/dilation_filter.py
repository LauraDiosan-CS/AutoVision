from copy import deepcopy

from objects.pipe_data import PipeData
from objects.video_info import VideoInfo
from vision_pipeline.filters.base_filter import BaseFilter
import cv2


class DilationFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, kernel_size=1):
        super().__init__(video_info=video_info)
        self.kernel_size = kernel_size

    def process(self, data: PipeData) -> PipeData:
        # old version: kernel = ones((self.kernel_size, self.kernel_size), uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        data.frame = cv2.dilate(data.frame, kernel, iterations=1)
        data.processed_frames.append(deepcopy(data.frame))
        return data