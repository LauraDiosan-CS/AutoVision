import cv2
from ultralytics import YOLO
import torch

from filters.base_filter import BaseFilter
from objects.pipe_data import PipeData
from objects.types.video_info import VideoInfo
from objects.types.road_info import RoadObject


class ObjectDetectionFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, model_path):
        super().__init__(video_info=video_info, visualize=visualize)
        self.model = YOLO(model_path)

        if torch.cuda.is_available():
            print(f'\n{model_path} running on gpu...\n')
        else:
            print(f'\n{model_path} running on cpu...\n')

        self.result = None

    def pre_process_result(self, yolo_results, data: PipeData) -> PipeData:
        pass

    def process(self, data: PipeData) -> PipeData:
        if torch.cuda.is_available():
            self.model.cuda()

        yolo_results = self.model(data.frame, verbose=False)
        data = self.pre_process_result(yolo_results[0], data)
        data.frame = yolo_results[0].plot()

        return super().process(data)


def get_distance_from_realsense(depth_frame, bbox_list):
    xscaling = 0.3333333333
    yscaling = 0.4444444444
    x = int((bbox_list[0] + bbox_list[2]) / 2 * xscaling)
    y = int((bbox_list[1] + bbox_list[3]) / 2 * yscaling)
    return depth_frame[y, x]


class SignsDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, model_path):
        super().__init__(video_info=video_info, visualize=visualize, model_path=model_path)

    def pre_process_result(self, yolo_results, data: PipeData) -> PipeData:
        labels = yolo_results.names

        for yolo_object in yolo_results:
            prediction_id = int(yolo_object.boxes.cls.item())
            prediction_label = labels[prediction_id]

            confidence = f'{yolo_object.boxes.conf.item():.2f}'

            bbox_tensor_cpu = yolo_object.boxes.xyxy.cpu()
            bbox_list = [float(f'{el:.4f}') for el in bbox_tensor_cpu.tolist()[0]]

            data.frame = data.frame.copy()
            cv2.circle(data.frame, (int((bbox_list[0] + bbox_list[2]) / 2), int((bbox_list[1] + bbox_list[3]) / 2)), 4,
                       (255, 0, 0), 5)

            if data.depth_frame is not None:  # check if realsense is connected and depth frame is available
                distance = get_distance_from_realsense(data.depth_frame, bbox_list)
            else:
                distance = float("inf")
            road_object = RoadObject(bbox=bbox_list, label=prediction_label, conf=confidence, distance=distance)

            data.traffic_signs.append(road_object)

        return data


class TrafficLightDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, model_path):
        super().__init__(video_info=video_info, visualize=visualize, model_path=model_path)

    def pre_process_result(self, yolo_results, data: PipeData) -> PipeData:
        return data


class PedestrianDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, model_path):
        super().__init__(video_info=video_info, visualize=visualize, model_path=model_path)

    def pre_process_result(self, yolo_results, data: PipeData) -> PipeData:
        return data