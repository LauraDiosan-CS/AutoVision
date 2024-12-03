from abc import ABC, abstractmethod

from perception.objects.pipe_data import PipeData
from perception.objects.video_info import VideoInfo, VideoRois


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
    def video_width(self):
        return self.video_info.width

    @property
    def video_height(self):
        return self.video_info.height

    @abstractmethod
    def process(self, data: PipeData) -> PipeData:
        """
        Abstract method that should be implemented by all filters
        :param data: PipeData object
        """
        # process the data

        if self.visualize:
            data.add_processed_frame(data.frame)

        return data