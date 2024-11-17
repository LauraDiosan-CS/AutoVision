from perception.filters.base_filter import BaseFilter
from perception.objects.pipe_data import PipeData
from perception.objects.video_info import VideoInfo

from cv2 import Canny

class CannyEdgeFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, low_threshold, high_threshold):
        super().__init__(video_info=video_info, visualize=visualize)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, data: PipeData) -> PipeData:
        data.frame = Canny(data.frame, self.low_threshold, self.high_threshold)

        return super().process(data)