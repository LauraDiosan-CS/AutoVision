import numpy as np
import math
from objects.pipe_data import PipeData
from objects.road_info import RoadMarkings, Line, Point
from objects.video_info import VideoInfo
import cv2
from vision_pipeline.filters.base_filter import BaseFilter
from helpers import calculate_angle_with_Ox


class LaneDetectFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo):
        super().__init__(video_info=video_info)
        self.center_line = []
        self.right_line = []

    def process(self, data: PipeData) -> PipeData:
        hough_lines = self.apply_houghLines(data.frame)
        if hough_lines is not None:
            # hough_lines = self.remove_roi_lines(hough_lines)

            # lane detection
            self.detect_max_lanes(hough_lines)
            self.extend_lanes()
            print('center:', self.center_line, 'right:', self.right_line)
            if (self.center_line and self.right_line) and (len(self.center_line) == 2 and len(self.right_line) == 2):
                data.road_markings = RoadMarkings(left_line=None,
                                                  center_line=Line(
                                                      upper_point=Point(x=self.center_line[0][0],
                                                                        y=self.center_line[0][1]),
                                                      lower_point=Point(x=self.center_line[1][0],
                                                                        y=self.center_line[1][1])
                                                  ),
                                                  right_line=Line(
                                                      upper_point=Point(x=self.right_line[0][0],
                                                                        y=self.right_line[0][1]),
                                                      lower_point=Point(x=self.right_line[1][0],
                                                                        y=self.right_line[1][1])
                                                  ))

        return data

    # -----------------------------------------------
    # Processing Methods  

    def remove_roi_lines(self, hough_lines):
        # roi endpoints order: BL, TL, TR, BR
        roi = self.video_info.video_roi_bbox
        roi_left_limit = []
        roi_left_limit = [[roi[1][0], roi[1][1], roi[0][0], roi[0][1]]]
        roi_right_limit = np.array([[roi[2][0], roi[2][1], roi[3][0], roi[3][1]]])

        idx = np.where(hough_lines == roi_left_limit)
        return hough_lines[~np.all(hough_lines == roi_left_limit, axis=(1, 2))]

    @staticmethod
    def apply_houghLines(frame, rho=1, theta=np.pi / 180, threshold=50, min_line_length=100, max_line_gap=550):
        return cv2.HoughLinesP(frame, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                               maxLineGap=max_line_gap)

    def max_lane_lengths(self, hough_lines):
        max_left_lane = 0
        max_right_lane = 0
        if hough_lines is not None:
            for line in hough_lines:
                for x1, y1, x2, y2 in line:
                    if x2 < self.width / 2:
                        if abs(y1 - y2) > max_left_lane:
                            max_left_lane = abs(y1 - y2)
                    elif x2 >= self.width / 2:
                        if abs(y1 - y2) > max_right_lane:
                            max_right_lane = abs(y1 - y2)
        return max_left_lane, max_right_lane

    def detect_max_lanes(self, hough_lines):
        max_left_lane, max_right_lane = self.max_lane_lengths(hough_lines)
        # resetting lanes
        self.center_line, self.right_line = [], []

        if hough_lines is not None:
            for line in hough_lines:
                is_horizontal = False
                for x1, y1, x2, y2 in line:
                    # checking  if the line is approximately vertical
                    _, deg = calculate_angle_with_Ox(line)

                    if abs(deg) < 5:
                        is_horizontal = True

                    coords = line[0]
                    if not ((abs(y1 - y2) < 50) or (abs(x1 - x2) < 100)):

                        if x2 < self.width / 2 and abs(
                                y1 - y2) == max_left_lane and not is_horizontal and not self.center_line:
                            for i in range(2):
                                self.center_line.append((coords[2 * i], coords[2 * i + 1]))

                        elif x2 > self.width / 2 and abs(
                                y1 - y2) == max_right_lane and not is_horizontal and not self.right_line:
                            for i in range(2):
                                self.right_line.append((coords[2 * i], coords[2 * i + 1]))

    def extend_lanes(self):
        # left and right lanes extended down until lower limit
        ext_lower_left = []
        ext_lower_right = []
        # left and right lanes extended up untill upper limit
        ext_upper_left = []
        ext_upper_right = []

        if self.center_line and self.right_line:
            # 1. extending both lanes downuntil the bottom of frame
            left_slope = (self.center_line[1][1] - self.center_line[0][1]) / (
                    self.center_line[1][0] - self.center_line[0][0])
            right_slope = (self.right_line[1][1] - self.right_line[0][1]) / (
                    self.right_line[1][0] - self.right_line[0][0])

            y_intercept_left = self.center_line[0][1] - left_slope * self.center_line[0][0]
            y_intercept_right = self.right_line[0][1] - right_slope * self.right_line[0][0]

            x_left = (self.height - y_intercept_left) / left_slope
            bottom_left = (int(x_left), self.height)

            x_right = (self.height - y_intercept_right) / right_slope
            bottom_right = (int(x_right), self.height)

            ext_lower_left.append(self.center_line[1])
            ext_lower_left.append(bottom_left)

            ext_lower_right.append(self.right_line[0])
            ext_lower_right.append(bottom_right)

            # 2. extending both lanes up until the start point of the minimum length lane
            left_distance = self.lane_distance(ext_lower_left)
            right_distance = self.lane_distance(ext_lower_right)

            if left_distance < right_distance:
                # left lane is smaller => extend it up until it reaches right lane
                minimum_startpoint = ext_lower_left[0]
                intersection_x_right = self.find_intersection_x_coordinate(right_slope, y_intercept_right,
                                                                           minimum_startpoint[1])

                for point in ext_lower_left:
                    ext_upper_left.append(point)

                ext_upper_right.append((intersection_x_right, minimum_startpoint[1]))
                ext_upper_right.append(ext_lower_right[1])

            else:
                # right lane is smaller => extend it up until it reaches left lane
                minimum_startpoint = ext_lower_right[0]
                intersection_x_left = self.find_intersection_x_coordinate(left_slope, y_intercept_left,
                                                                          minimum_startpoint[1])

                for point in ext_lower_right:
                    ext_upper_right.append(point)

                ext_upper_left.append((intersection_x_left, minimum_startpoint[1]))
                ext_upper_left.append(ext_lower_left[1])

        self.center_line = ext_upper_left
        self.right_line = ext_upper_right

    def lane_distance(self, lane_endpoints):
        x1 = lane_endpoints[0][0]
        y1 = lane_endpoints[0][1]
        x2 = lane_endpoints[1][0]
        y2 = lane_endpoints[1][1]

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def find_intersection_x_coordinate(self, slope, y_intercept, y_horizontal_line):
        return int((y_horizontal_line - y_intercept) / slope)