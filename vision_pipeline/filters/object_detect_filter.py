from ultralytics import YOLO
import torch
from objects.pipe_data import PipeData
from objects.video_info import VideoInfo
from vision_pipeline.filters.base_filter import BaseFilter


class ObjectDetectionFilter(BaseFilter):
    def __init__(self,videoInfo: VideoInfo, model):
        super().__init__(video_info=videoInfo)
        self.model = YOLO(model)

    def process(self, data: PipeData) -> PipeData:
        if torch.cuda.is_available():
            print('running on gpu...')
            print(data.frame.shape)
            self.model.cuda()
        else:
            print('running on cpu...')
        yolo_results = self.model(data.frame)
        data.frame = yolo_results[0].plot()
        return data


class SignsDetect(ObjectDetectionFilter):
    def __init__(self,videoInfo:VideoInfo, model):
        super().__init__(videoInfo, model)


class TrafficLightDetect(ObjectDetectionFilter):
    def __init__(self,videoInfo:VideoInfo, model):
        super().__init__(videoInfo, model)


class PedestrianDetect(ObjectDetectionFilter):
    def __init__(self,videoInfo:VideoInfo, model):
        super().__init__(videoInfo, model)