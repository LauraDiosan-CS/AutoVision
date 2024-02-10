from abc import ABC, abstractmethod

from objects.pipe_data import PipeData
from objects.video_info import VideoInfo


class BaseFilter(ABC):
    def __init__(self, video_info: VideoInfo):
        self.video_info = video_info

    @property
    def video_name(self):
        return self.video_info.video_name

    @property
    def video_roi_bbox(self):
        return self.video_info.video_roi_bbox

    @property
    def width(self):
        return self.video_info.width

    @property
    def height(self):
        return self.video_info.height

    @abstractmethod
    def process(self, data: PipeData) -> PipeData:
        pass