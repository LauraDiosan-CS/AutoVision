import numpy as np
import math
from objects.pipe_data import PipeData
from objects.types.road_info import RoadMarkings, Line, Point
from objects.types.video_info import VideoInfo
import cv2
from filters.base_filter import BaseFilter
import math


def calculate_angle_with_Ox(line):
    for x1, y1, x2, y2 in line:
        delta_x = x2 - x1
        delta_y = y2 - y1
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
        return angle_rad, angle_deg
    raise ValueError("Line is empty")


class LaneDetectFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool):
        super().__init__(video_info=video_info, visualize=visualize)
        self.center_line = []
        self.right_line = []
        self.stop_lines = []

    def process(self, data: PipeData) -> PipeData:
        hough_lines = self.apply_houghLines(data.frame)

        if hough_lines is not None:
            # lane detection
            # print('\n lines:', len(hough_lines))
            self.detect_max_lanes(hough_lines)
            # print('max center:', self.center_line)
            # print('max right:', self.right_line)
            self.extend_lanes()
            # print('ext center:', self.center_line)
            # print('ext right:', self.right_line)
            # print('center:', self.center_line, 'right:', self.right_line)
                
            if (self.center_line and self.right_line) and (len(self.center_line) == 2 and len(self.right_line) == 2):
                horizontals, right_intersections, center_intersections = self.detect_horizontals(hough_lines, self.center_line, self.right_line)

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
                                                  ),
                                                  stop_lines= [
                                                      Line( 
                                                      upper_point=Point(x=stop_line[0][0],
                                                                        y=stop_line[0][1]),
                                                      lower_point=Point(x=stop_line[1][0],
                                                                        y=stop_line[1][1])
                                                      ) 
                                                      for stop_line in self.stop_lines
                                                  ],
                                                  horizontals=horizontals,
                                                  right_int=right_intersections,
                                                  center_int=center_intersections
                                                  )
                
                data.frame = cv2.cvtColor(data.frame, cv2.COLOR_GRAY2BGR)
                # print('center:', data.road_markings.center_line)

        return super().process(data)

    # -----------------------------------------------
    # Processing Methods
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

    def distances(self):
        '''The distance between 2 lines is used to filter out the horizontal lines that appear clustered in the same area
        and it's measured by taking a point from one line and calculate it's perpendicular onto the second line'''

        #print('\ndistances:')
        if len(self.stop_lines):
            line1 = self.stop_lines[0]
            slope1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0])
            y_intercept_1 = line1[0][1] - slope1 * line1[0][0]

            for line2 in self.stop_lines[1:]:
                slope2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0])
                y_intercept_2 = line2[0][1] - slope2 * line2[0][0]

                dist = abs(-y_intercept_1 + y_intercept_2) / math.sqrt(slope2 ** 2 + 1)

                y_intercept_1 = y_intercept_2
                if dist < 500:
                    self.stop_lines.remove(line2)
                else:
                    #print('dist:', dist)
                    pass

        
    def detect_horizontals(self, hough_lines, center_line, right_line):
        stop_lines = []
        horizontals = []
        right_intersections = []
        center_intersections = []

        if hough_lines is not None:
            for line in hough_lines:
                _, deg = calculate_angle_with_Ox(line)
                if abs(deg) < 3:
                    horizontals.append(line)

        for line in horizontals:
            #print('bef:', line)
           # print('\nline:', line)
            # slope = (y2 - y1)/(x2 - x1)
            slope_center = (center_line[1][1] - center_line[0][1]) / (center_line[1][0] - center_line[0][0])
            slope_right = (right_line[1][1] - right_line[0][1]) / (right_line[1][0] - right_line[0][0])
            slope_line = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])

            # y_intercept = y - slope * x   
            y_intercept_center = center_line[0][1] - slope_center * center_line[0][0]
            y_intercept_right = right_line[0][1] - slope_right * right_line[0][0]
            y_intercept_line = line[0][1] - slope_line * line[0][0]

            # x_intersect = (y_intercept2 -y_intercept1) / (slope1 - slope2)
            x_intersect_center = (y_intercept_center - y_intercept_line) / (slope_line - slope_center)
            x_intersect_right = (y_intercept_right - y_intercept_line) / (slope_line - slope_right)

            # y_intersect = slope * x_intersect + y_intercept
            y_intersect_center = slope_center * x_intersect_center + y_intercept_center
            y_intersect_right = slope_right * x_intersect_right + y_intercept_right

            intersect_center = (int(x_intersect_center), int(y_intersect_center))
            intersect_right = (int(x_intersect_right), int(y_intersect_right))

            # print('center:', intersect_center)
            # print('right:', intersect_right)
            
            right_intersections.append(intersect_right)
            center_intersections.append(intersect_center)

            horizontal_line = (intersect_center, intersect_right)
            #print('aft:', horizontal_line)
            stop_lines.append(horizontal_line)
        
        #print('stop lines:', self.stop_lines)
        self.stop_lines = stop_lines
        self.distances()
        return horizontals, right_intersections, center_intersections

    def detect_max_lanes(self, hough_lines):
        self.center_line, self.right_line = [], []
        max_left_lane = 0
        max_right_lane = 0

        if hough_lines is not None:
            for line in hough_lines:

                _, deg = calculate_angle_with_Ox(line)

                is_horizontal = False
                if abs(deg) < 5:
                    is_horizontal = True

                coords = line[0]

                for x1, y1, x2, y2 in line:
                    if x2 < self.width / 2 and abs(y1 - y2) >= max_left_lane and not is_horizontal:
                        max_left_lane = abs(y1 - y2)
                        for i in range(2):
                            self.center_line.append((coords[2 * i], coords[2 * i + 1]))
                    elif x2 >= self.width / 2 and abs(y1 - y2) >= max_right_lane and not is_horizontal:
                        max_right_lane = abs(y1 - y2)
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
        # slope, y_intercept, startpoint - of the same singular line

        return int((y_horizontal_line - y_intercept) / slope)