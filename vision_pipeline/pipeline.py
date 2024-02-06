from copy import deepcopy
from tracemalloc import BaseFilter
from video_info import VideoInfo
from vision_pipeline.filters.basic_filters.blur_filter import BlurFilter
from vision_pipeline.filters.basic_filters.cannyedge_filter import CannyEdgeFilter
from vision_pipeline.filters.basic_filters.dilation_filter import DilationFilter
from vision_pipeline.filters.basic_filters.grayscale_filter import GrayScaleFilter
from vision_pipeline.filters.basic_filters.roi_filter import ROIFilter
from vision_pipeline.filters.draw_filter import DrawFilter

from vision_pipeline.filters.lane_detect_filter import LaneDetectFilter
from vision_pipeline.filters.object_detect_filter import SignsDetect
from fastcore.parallel import parallel
import numpy as np
from vision_pipeline.filters.steer import Steer


class Pipeline:
    def __init__(self, args, video_info: VideoInfo):
        self.signs_model = args.yolo_signs
        self.roi_filter = ROIFilter(video_info=video_info)
        self.grayscale_filter = GrayScaleFilter(video_info=video_info)
        self.blur_filter = BlurFilter(video_info=video_info)
        self.canny_edge_filter = CannyEdgeFilter(video_info=video_info)
        self.dilation_filter = DilationFilter(video_info=video_info)
        self.lane_detection_filter = LaneDetectFilter(video_info=video_info)
        self.steer_filter = Steer(video_info=video_info)
        self.sign_detection_filter = SignsDetect(videoInfo=video_info,model=self.signs_model)
        self.lane_detection_configuration: list[BaseFilter] = [self.roi_filter, self.grayscale_filter, 
                                                               self.canny_edge_filter, self.lane_detection_filter, self.steer_filter]
        self.sign_detection_configuration: list[BaseFilter] = [self.sign_detection_filter]
        self.parallel_configurations: list[list[BaseFilter]] = [self.sign_detection_configuration, self.lane_detection_configuration]

    # sequential processing of the pipeline
    def run_seq(self, frame):
        data = frame
        processed_frames = []
        steering_angle = None
        road_markings = None
        for config in self.parallel_configurations:
            for filter in config:
                data = filter.process(data)
                match filter.return_type:
                    case "steering_angle":
                        steering_angle = data
                    case "img":
                        processed_frames.append(deepcopy(data))
                    case "RoadMarkings":
                        road_markings = data

        return processed_frames, steering_angle, road_markings #yolo stuff

    # parallel processing of the pipeline
    # def run_par(self, frame):
    #     # it's very slow for some reason =)) can not figure it out
    #     processing_pairs = [(frame, filter) for filter in self.parallel_filters]
    #     processed_frames = parallel(self.apply_filter, processing_pairs, n_workers=2)

    #     combined_frame = np.maximum(*processed_frames)
    #     processed_frames.append(combined_frame)

    #     return processed_frames

    # def apply_filter(self, processing_pairs):
    #     frame, filter = processing_pairs
    #     return filter(frame)