from perception.filters.base_filter import BaseFilter
from perception.objects.pipe_data import PipeData
from perception.objects.video_info import VideoInfo
from cv2 import cvtColor, COLOR_BGR2GRAY


class GrayScaleFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool):
        super().__init__(video_info=video_info, visualize=visualize)

    def process(self, data: PipeData) -> PipeData:
        data.frame = cvtColor(data.frame, COLOR_BGR2GRAY)

        return super().process(data)