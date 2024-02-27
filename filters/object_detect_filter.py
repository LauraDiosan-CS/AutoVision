from ultralytics import YOLO
import torch

from filters.base_filter import BaseFilter
from objects.pipe_data import PipeData
from objects.types.video_info import VideoInfo
from objects.types.road_info import RoadObject


class ObjectDetectionFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info=video_info)
        self.model = YOLO(model_path)
        self.result = None
    
    def pre_process_result(self, result):
        pass

    def process(self, data: PipeData) -> PipeData:
        if torch.cuda.is_available():
            print('running on gpu...')
            print(data.frame.shape)
            self.model.cuda()
        else:
            print('running on cpu...')
        
        yolo_results = self.model(data.frame)
        self.pre_process_result(yolo_results[0], data)
        data.frame = yolo_results[0].plot()
        data.processed_frames.append(data.frame.copy())
        return data
    

class SignsDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)

    def pre_process_result(self, result, data):
        data.traffic_signs = []
        labels = result.names
        print('\nresult:', labels)
        for object in result:
            prediction_id = int(object.boxes.cls.item())
            prediction_label = labels[prediction_id]
            confidence = f'{object.boxes.conf.item():.2f}'
            bbox_tensor_cpu = object.boxes.xyxy.cpu()
            bbox_list = [el for el in bbox_tensor_cpu.tolist()]
            print('label:', prediction_label)
            print('conf:', confidence)
            print('bbox:', object.boxes.xyxy, bbox_list)


class TrafficLightDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)


class PedestrianDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)