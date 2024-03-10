from abc import ABC, abstractmethod

from objects.pipe_data import PipeData
from objects.types.video_info import VideoRois, VideoInfo


class BaseFilter(ABC):
    def __init__(self, video_info: VideoInfo, visualize: bool):
        self.video_info = video_info
        self.visualize = visualize

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
        """
        Abstract method that should be implemented by all filters
        :param data: PipeData object
        """
        # process the data

        if self.visualize:
            data.processed_frames.append(data.frame.copy())

        return data
