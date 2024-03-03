from objects.pipe_data import PipeData
from filters.base_filter import BaseFilter
import cv2

from objects.types.video_info import VideoInfo


class DilationFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, kernel_size, iterations):
        super().__init__(video_info=video_info, visualize=visualize)
        self.kernel_size = kernel_size
        self.iterations = iterations

    def process(self, data: PipeData) -> PipeData:
        # old version: kernel = ones((self.kernel_size, self.kernel_size), uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        data.frame = cv2.dilate(data.frame, kernel, iterations=self.iterations)

        return super().process(data)