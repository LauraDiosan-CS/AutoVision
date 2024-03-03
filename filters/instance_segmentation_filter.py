from ultralytics import YOLO
import torch

from filters.base_filter import BaseFilter
from objects.pipe_data import PipeData
from objects.types.video_info import VideoInfo

class YOLOInstanceSegmentationFilter(BaseFilter):
    def __init__(self, video_info:VideoInfo, model_path):
        super().__init__(video_info=video_info)
        self.model = YOLO(model_path)
        self.result = None
        self.classes = []

    def process(self, data: PipeData) -> PipeData:
        if torch.cuda.is_available():
            print('running on gpu...')
            self.model.cuda()
        else:
            print('running on cpu...')
        
        yolo_seg_results = self.model(data.frame, classes=self.classes)
        data.frame = yolo_seg_results[0].plot()
        data.processed_frames.append(data.frame.copy())
        return data
    
class CarSegment(YOLOInstanceSegmentationFilter):
    def __init__(self, video_info: VideoInfo, model_path):
        super().__init__(video_info, model_path)
        self.classes = [2]
    


        

