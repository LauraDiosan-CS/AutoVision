from collections import namedtuple
from typing import Union

from filters.base_filter import BaseFilter
from filters.basic_filters.blur_filter import BlurFilter
from filters.basic_filters.cannyedge_filter import CannyEdgeFilter
from filters.basic_filters.dilation_filter import DilationFilter
from filters.basic_filters.grayscale_filter import GrayScaleFilter
from filters.roi_filter import ROIFilter
from filters.object_detect_filter import SignsDetect
from filters.lane_detect_filter import LaneDetectFilter
from filters.heading_error_filter import HeadingErrorFilter
from filters.object_detect_filter import TrafficLightDetect
from filters.object_detect_filter import PedestrianDetect

JSONPipelineConfig = list[dict[str, dict[str, Union[int, str, float]]]]
PipelineConfig = list[list[BaseFilter]]

FilterClassWithParams = namedtuple("FilterClassWithParams", ["filter_class", "expected_params"])

FILTER_CLASS_LOOKUP: dict[str:FilterClassWithParams] = {
    "blur": FilterClassWithParams(BlurFilter, ["kernel_size", "sigmaX"]),
    "dilation": FilterClassWithParams(DilationFilter, ["kernel_size", "iterations"]),
    "grayscale": FilterClassWithParams(GrayScaleFilter, []),
    "canny_edge": FilterClassWithParams(CannyEdgeFilter, ["low_threshold", "high_threshold"]),
    "roi": FilterClassWithParams(ROIFilter, ["roi_type"]),
    "lane_detect": FilterClassWithParams(LaneDetectFilter, []),
    "heading_error": FilterClassWithParams(HeadingErrorFilter, []),
    "signs_detect": FilterClassWithParams(SignsDetect, ["model_path"]),
    "traffic_light_detect": FilterClassWithParams(TrafficLightDetect, ["model_path"]),
    "pedestrian_detect": FilterClassWithParams(PedestrianDetect, ["model_path"])
}