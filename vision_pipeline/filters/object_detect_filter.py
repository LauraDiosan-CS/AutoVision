from ultralytics import YOLO
import torch
from video_info import VideoInfo
from vision_pipeline.filters.base_filter import BaseFilter


class ObjectDetectionFilter(BaseFilter):
    def __init__(self,videoInfo: VideoInfo, model):
        super().__init__(video_info=videoInfo, return_type="img")
        self.model = YOLO(model)

    def process(self, frame):
        if torch.cuda.is_available():
            print('running on gpu...')
            print(frame.shape)
            self.model.cuda()
        else:
            print('running on cpu...')
        yolo_results = self.model(frame)
        return yolo_results[0].plot()


class SignsDetect(ObjectDetectionFilter):
    def __init__(self,videoInfo:VideoInfo, model):
        super().__init__(videoInfo, model)


class TrafficLightDetect(ObjectDetectionFilter):
    def __init__(self,videoInfo:VideoInfo, model):
        super().__init__(videoInfo, model)


class PedestrianDetect(ObjectDetectionFilter):
    def __init__(self,videoInfo:VideoInfo, model):
        super().__init__(videoInfo, model)