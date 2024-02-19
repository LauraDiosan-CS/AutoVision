from ultralytics import YOLO
import torch

from filters.base_filter import BaseFilter
from objects.pipe_data import PipeData
from objects.types.video_info import VideoInfo


class ObjectDetectionFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info=video_info)
        self.model = YOLO(model_path)

    def process(self, data: PipeData) -> PipeData:
        if torch.cuda.is_available():
            print('running on gpu...')
            print(data.frame.shape)
            self.model.cuda()
        else:
            print('running on cpu...')
        yolo_results = self.model(data.frame)
        data.frame = yolo_results[0].plot()
        data.processed_frames.append(data.frame.copy())
        return data


class SignsDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)


class TrafficLightDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)


class PedestrianDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)