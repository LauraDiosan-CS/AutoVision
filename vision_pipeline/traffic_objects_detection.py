from ultralytics import YOLO
import torch


class YoloDetection():
    def __init__(self, model):
        self.model = YOLO(model)

    def process(self, frame):
        if torch.cuda.is_available():
            print('running on gpu...')
            self.model.cuda()
        else:
            print('running on cpu...')
        yolo_results = self.model(frame)
        return yolo_results[0].plot()


class SignsDetect(YoloDetection):
    def __init__(self, model):
        super().__init__(model)


class TrafficLightDetect(YoloDetection):
    def __init__(self, model):
        super().__init__(model)


class PedestrianDetect(YoloDetection):
    def __init__(self, model):
        super().__init__(model)