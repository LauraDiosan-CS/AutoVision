import cv2
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
        print('signs:', data.traffic_signs)
        data.frame = yolo_results[0].plot()
        data.processed_frames.append(data.frame.copy())
        return data
    

class SignsDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)

    def get_distance_from_realsense(self, frame, bbox_list):

        if frame is None:
            return 0
        else:
            xscaling = 0.3333333333
            yscaling = 0.4444444444
            x=int((bbox_list[0]+bbox_list[2])/2*xscaling)
            y=int((bbox_list[1]+bbox_list[3])/2*yscaling)
            print('x', x)
            print('y', y)
            print('data', frame[y,x])
        return frame[y,x]


    def pre_process_result(self, result, data):
        labels = result.names
        print('\nresult:', labels)

        for object in result:
            prediction_id = int(object.boxes.cls.item())
            prediction_label = labels[prediction_id]

            confidence = f'{object.boxes.conf.item():.2f}'

            bbox_tensor_cpu = object.boxes.xyxy.cpu()
            bbox_list = [float(f'{el:.4f}') for el in bbox_tensor_cpu.tolist()[0]]

            distance = self.get_distance_from_realsense(data.depth_frame, bbox_list)

            cv2.circle(data.frame, (int((bbox_list[0]+bbox_list[2])/2), int((bbox_list[1]+bbox_list[3])/2)),4,(255,0,0), 5)

            road_object = RoadObject(bbox=bbox_list, label=prediction_label, conf=confidence, distance=distance)
            
            data.traffic_signs.append(road_object)

            print('\ntypes:')
            print('bbox:', type(bbox_list), bbox_list)
            print('conf:', type(confidence))
            print('label:', type(prediction_label))
            print("ro", road_object)

        return data

            
class TrafficLightDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)
    
    def pre_process_result(self, result, data):
        return data


class PedestrianDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)
    
    def pre_process_result(self, result, data):
        return data