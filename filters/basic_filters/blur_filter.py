from objects.pipe_data import PipeData
from filters.base_filter import BaseFilter
import cv2

from objects.types.video_info import VideoInfo


class BlurFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo,visualize: bool, kernel_size, sigmaX):
        super().__init__(video_info=video_info, visualize=visualize)
        self.kernel_size = kernel_size
        self.sigmaX = sigmaX

    def process(self, data: PipeData) -> PipeData:
        data.frame = cv2.GaussianBlur(data.frame, (self.kernel_size, self.kernel_size), self.sigmaX)

        return super().process(data)