from copy import deepcopy
from objects.pipe_data import PipeData
from objects.video_info import VideoInfo
from vision_pipeline.filters.base_filter import BaseFilter
from vision_pipeline.filters.basic_filters.blur_filter import BlurFilter
from vision_pipeline.filters.basic_filters.cannyedge_filter import CannyEdgeFilter
from vision_pipeline.filters.basic_filters.dilation_filter import DilationFilter
from vision_pipeline.filters.basic_filters.grayscale_filter import GrayScaleFilter
from vision_pipeline.filters.basic_filters.roi_filter import ROIFilter
from vision_pipeline.filters.draw_filter import DrawFilter

from vision_pipeline.filters.lane_detect_filter import LaneDetectFilter
from vision_pipeline.filters.object_detect_filter import SignsDetect
from vision_pipeline.filters.heading_error_filter import HeadingErrorFilter


class Pipeline:
    def __init__(self, args, video_info: VideoInfo):
        self.signs_model = args.yolo_signs

        self.roi_filter = ROIFilter(video_info=video_info)
        self.grayscale_filter = GrayScaleFilter(video_info=video_info)
        self.blur_filter = BlurFilter(video_info=video_info)
        self.canny_edge_filter = CannyEdgeFilter(video_info=video_info)
        self.dilation_filter = DilationFilter(video_info=video_info)
        self.lane_detection_filter = LaneDetectFilter(video_info=video_info)
        self.heading_error_filter = HeadingErrorFilter(video_info=video_info)
        self.sign_detection_filter = SignsDetect(videoInfo=video_info, model=self.signs_model)
        self.draw_filter = DrawFilter(video_info=video_info)

        self.lane_detection_configuration: list[BaseFilter] = [self.grayscale_filter, self.canny_edge_filter, self.blur_filter, self.dilation_filter, self.roi_filter,
                                                               self.lane_detection_filter,self.heading_error_filter, self.draw_filter]
        self.sign_detection_configuration: list[BaseFilter] = [self.sign_detection_filter]

        self.parallel_configurations: list[list[BaseFilter]] = [self.lane_detection_configuration, self.sign_detection_configuration]

    def run_seq(self, frame, frame_ct):
        data: PipeData = PipeData(frame=frame, road_markings=None, steering_angle=None, unfiltered_frame=deepcopy(frame))

        for config in self.parallel_configurations:
            for filter in config:
                data = filter.process(data)

        return data  # TODO add yolo stuff