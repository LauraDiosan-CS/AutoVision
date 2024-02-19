from copy import deepcopy

from objects.pipe_data import PipeData
from filters.base_filter import BaseFilter
import cv2

from objects.types.video_info import VideoInfo


class GrayScaleFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo):
        super().__init__(video_info=video_info)

    def process(self, data: PipeData) -> PipeData:
        data.frame = cv2.cvtColor(data.frame, cv2.COLOR_BGR2GRAY)
        data.processed_frames.append(deepcopy(data.frame))
        return data