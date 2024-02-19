from abc import ABC, abstractmethod

from objects.pipe_data import PipeData
from objects.types.video_info import VideoRois, VideoInfo


class BaseFilter(ABC):
    def __init__(self, video_info: VideoInfo):
        self.video_info = video_info

    @property
    def video_name(self):
        return self.video_info.video_name

    @property
    def video_rois(self) -> VideoRois:
        return self.video_info.video_rois

    @property
    def width(self):
        return self.video_info.width

    @property
    def height(self):
        return self.video_info.height

    @abstractmethod
    def process(self, data: PipeData) -> PipeData:
        pass