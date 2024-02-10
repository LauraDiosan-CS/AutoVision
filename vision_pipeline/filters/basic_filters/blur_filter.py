from copy import deepcopy

from objects.pipe_data import PipeData
from objects.video_info import VideoInfo
from vision_pipeline.filters.base_filter import BaseFilter
import cv2


class BlurFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, kernel_size=7, sigmaX=1):
        super().__init__(video_info=video_info)
        self.kernel_size = kernel_size
        self.sigmaX = sigmaX

    def process(self, data: PipeData) -> PipeData:
        data.frame = cv2.GaussianBlur(data.frame, (self.kernel_size, self.kernel_size), self.sigmaX)
        data.processed_frames.append(deepcopy(data.frame))
        return data