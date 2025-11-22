import cv2
from ultralytics import YOLO
import torch

from perception.filters.base_filter import BaseFilter
from perception.objects.pipe_data import PipeData
from perception.objects.road_info import RoadObject
from perception.objects.video_info import VideoInfo


class ObjectDetectionFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, model_path, verbose):
        super().__init__(video_info=video_info, visualize=visualize)
        self.model = YOLO(model_path)
        self.verbose = verbose

        if torch.cuda.is_available():
            print(f'{model_path} running on gpu...')
        else:
            print(f'{model_path} running on cpu...')

        self.result = None

    def pre_process_result(self, yolo_results, data: PipeData, confidence_threshold=0.5):
        labels = yolo_results.names
        results = []

        for yolo_object in yolo_results:
            prediction_id = int(yolo_object.boxes.cls.item())
            prediction_label = labels[prediction_id]

            confidence = round(float(yolo_object.boxes.conf.item()), 2)

            if confidence < confidence_threshold:
                continue
            bbox_tensor_cpu = yolo_object.boxes.xyxy.cpu()
            bbox_list = [float(f'{el: .4f}') for el in bbox_tensor_cpu.tolist()[0]]

            if self.visualize:
                data.frame = data.frame.copy() # make it writable
                cv2.circle(data.frame, (int((bbox_list[0] + bbox_list[2]) / 2), int((bbox_list[1] + bbox_list[3]) / 2)), 4,
                           (255, 0, 0), 5)
            if data.depth_frame is not None:  # check if realsense is connected and depth frame is available
                distance = get_distance_from_realsense(data.depth_frame, bbox_list)
            else:
                distance = float("inf")
            road_object = RoadObject(bbox=bbox_list, label=prediction_label, conf=confidence, distance=distance)
            results.append(road_object)

        return sorted(results, key=lambda x: x.distance)

    def process(self, data):
        return super().process(data)


def get_distance_from_realsense(depth_frame, bbox_list):
    xscaling = 640 / 1280
    yscaling = 480 / 720
    x = int((bbox_list[0] + bbox_list[2]) / 2 * xscaling)
    y = int((bbox_list[1] + bbox_list[3]) / 2 * yscaling)
    return depth_frame[y, x]


class SignsDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, model_path, verbose=False):
        super().__init__(video_info=video_info, visualize=visualize, model_path=model_path, verbose=verbose)

    def process(self, data):
        if torch.cuda.is_available():
            self.model.cuda()

        yolo_results = self.model(data.frame, verbose=self.verbose)
        data.traffic_signs = self.pre_process_result(yolo_results[0], data, 0.3)
        data.frame = yolo_results[0].plot()

        return super().process(data)


class TrafficLightDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, model_path, verbose=False):
        super().__init__(video_info=video_info, visualize=visualize, model_path=model_path, verbose=verbose)

    def process(self, data):
        if torch.cuda.is_available():
            self.model.cuda()

        yolo_results = self.model(data.frame, verbose=self.verbose)
        data.traffic_lights = self.pre_process_result(yolo_results[0], data, 0.3)
        data.frame = yolo_results[0].plot()

        return super().process(data)


class PedestrianDetect(ObjectDetectionFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool, model_path, verbose=False):
        super().__init__(video_info=video_info, visualize=visualize, model_path=model_path, verbose=verbose)

    def process(self, data):
        if torch.cuda.is_available():
            self.model.cuda()

        yolo_results = self.model(data.frame, verbose=self.verbose)
        data.pedestrians = self.pre_process_result(yolo_results[0], data, 0.3)
        data.frame = yolo_results[0].plot()

        return super().process(data)