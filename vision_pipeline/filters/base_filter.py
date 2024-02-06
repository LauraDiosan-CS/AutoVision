from abc import ABC, abstractmethod

from video_info import VideoInfo

class BaseFilter(ABC):
    def __init__(self, video_info: VideoInfo, return_type: str):
        self.video_info = video_info
        self.return_type = return_type

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
    def process(self, frame):
        pass