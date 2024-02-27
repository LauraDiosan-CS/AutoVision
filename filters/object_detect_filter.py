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
        data = self.pre_process_result(yolo_results[0], data)
        data.frame = yolo_results[0].plot()
        data.processed_frames.append(data.frame.copy())
        return data
    

class SignsDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)

    def get_distance_from_realsense(self, frame):
        pass

    def pre_process_result(self, result, data):
        traffic_signs = []
        labels = result.names
        print('\nresult:', labels)

        for object in result:
            prediction_id = int(object.boxes.cls.item())
            prediction_label = labels[prediction_id]

            confidence = f'{object.boxes.conf.item():.2f}'

            bbox_tensor_cpu = object.boxes.xyxy.cpu()
            bbox_list = [float(f'{el:.4f}') for el in bbox_tensor_cpu.tolist()[0]]
            bbox_list = []

            distance = self.get_distance_from_realsense(data.depth_frame)

            road_object = RoadObject(bbox= bbox_list, label= prediction_label,
                                     conf= confidence, distance=None)
            
            traffic_signs.append(road_object)

            print('\ntypes:')
            print('bbox:', type(bbox_list), bbox_list)
            print('conf:', type(confidence))
            print('label:', type(prediction_label))
            print("ro", road_object)

        data.traffic_signs = traffic_signs
        return data

            
class TrafficLightDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)


class PedestrianDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)