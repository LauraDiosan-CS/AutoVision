from dataclasses import dataclass
from typing import Type, List, Union

from perception.filters.base_filter import BaseFilter
from perception.filters.basic_filters.blur_filter import BlurFilter
from perception.filters.basic_filters.cannyedge_filter import CannyEdgeFilter
from perception.filters.basic_filters.dilation_filter import DilationFilter
from perception.filters.basic_filters.grayscale_filter import GrayScaleFilter
from perception.filters.roi_filter import ROIFilter
from perception.filters.object_detect_filter import SignsDetect
from perception.filters.lane_detect_filter import LaneDetectFilter
from perception.filters.heading_error_filter import HeadingErrorFilter
from perception.filters.object_detect_filter import TrafficLightDetect
from perception.filters.object_detect_filter import PedestrianDetect

json_filters_class_params_type = dict[str, Union[str, int, bool]]
json_filters_type = dict[str, json_filters_class_params_type]

# its either the "name" (str) or the "filters" (json_filters_type)
JSONPipelinesTYPE = list[dict[str, str | json_filters_type]]

@dataclass(slots=True)
class PipelineConfig:
    name: str
    filters: List[BaseFilter]

@dataclass(slots=True)
class FilterClassWithExpectedParams:
    filter_class: Type[BaseFilter]  # Reference to the filter class itself
    expected_params: list[str]  # List of strings containing the expected parameters

FILTER_CLASS_LOOKUP: dict[str, FilterClassWithExpectedParams] = {
    "blur": FilterClassWithExpectedParams(BlurFilter, ["visualize", "kernel_size", "sigmaX"]),
    "dilation": FilterClassWithExpectedParams(DilationFilter, ["visualize", "kernel_size", "iterations"]),
    "grayscale": FilterClassWithExpectedParams(GrayScaleFilter, ["visualize"]),
    "canny_edge": FilterClassWithExpectedParams(CannyEdgeFilter, ["visualize", "low_threshold", "high_threshold"]),
    "roi": FilterClassWithExpectedParams(ROIFilter, ["visualize", "roi_type"]),
    "lane_detect": FilterClassWithExpectedParams(LaneDetectFilter, ["visualize"]),
    "heading_error": FilterClassWithExpectedParams(HeadingErrorFilter, ["visualize"]),
    "signs_detect": FilterClassWithExpectedParams(SignsDetect, ["visualize", "model_path"]),
    "traffic_light_detect": FilterClassWithExpectedParams(TrafficLightDetect, ["visualize", "model_path"]),
    "pedestrian_detect": FilterClassWithExpectedParams(PedestrianDetect, ["visualize", "model_path"]),
}