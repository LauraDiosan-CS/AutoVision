import numpy as np
from objects.pipe_data import PipeData
from objects.types.line_segment import LineSegment
from objects.types.road_info import RoadMarkings
from objects.types.video_info import VideoInfo
import cv2
from filters.base_filter import BaseFilter


def filter_lane_by_type(hough_line_segments: list[LineSegment], frame_width: int):
    horizontal_lines = []
    left_lane_lines = []
    right_lane_lines = []
    for line_segment in hough_line_segments:
        if line_segment.check_is_horizontal(15):
            horizontal_lines.append(line_segment)
        else:
            if line_segment.upper_x < frame_width / 2:
                left_lane_lines.append(line_segment)
            else:
                right_lane_lines.append(line_segment)

    return left_lane_lines, right_lane_lines, horizontal_lines


class LaneDetectFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool):
        super().__init__(video_info=video_info, visualize=visualize)

    def process(self, data: PipeData) -> PipeData:
        hough_lines = self.apply_houghLines(data.frame)

        if hough_lines is not None:
            hough_line_segments = [LineSegment(*line[0]) for line in hough_lines]
            left_lane_lines, right_lane_lines, horizontal_lines = filter_lane_by_type(hough_line_segments, self.width)

            left_line_segment = None
            right_line_segment = None

            if len(left_lane_lines) > 0:
                left_line_segment = max(left_lane_lines, key=lambda l: l.compute_vertical_distance())

            if len(right_lane_lines) > 0:
                right_line_segment = max(right_lane_lines, key=lambda l: l.compute_euclidean_distance())

            if left_line_segment:
                left_line_segment = self.extend_line(left_line_segment)
            if right_line_segment:
                right_line_segment = self.extend_line(right_line_segment)

            data.road_markings = RoadMarkings(left_line=None,
                                              center_line=left_line_segment,
                                              right_line=right_line_segment,
                                              stop_lines=None,
                                              horizontals=None,
                                              right_int=None,
                                              center_int=None
                                              )
            if left_line_segment and right_line_segment:
                # horizontals, right_intersections, center_intersections = self.detect_horizontals(hough_lines,
                #                                                                                  self.center_line,
                #                                                                                  self.right_line)

                data.frame = cv2.cvtColor(data.frame, cv2.COLOR_GRAY2BGR)
            if self.visualize:
                frame = data.frame.copy()
                for line in horizontal_lines:
                    cv2.line(frame, line.lower_point, line.upper_point, (0, 0, 255), 1)

                for line in left_lane_lines:
                    cv2.line(frame, line.lower_point, line.upper_point, (255, 0, 0), 1)

                for line in right_lane_lines:
                    cv2.line(frame, line.lower_point, line.upper_point, (0, 255, 0), 1)
                data.add_processed_frame(frame)
                return data  # skip visualization from base filter

        return super().process(data)

    # -----------------------------------------------
    # Processing Methods
    @staticmethod
    def apply_houghLines(frame, rho=1, theta=np.pi / 180, threshold=50, min_line_length=300, max_line_gap=200):
        return cv2.HoughLinesP(frame, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                               maxLineGap=max_line_gap)

    # def distances(self):
    #     """The distance between 2 lines is used to filter out the horizontal lines that appear
    #         clustered in the same area, and it's measured by taking a point from one line and
    #         calculate it's perpendicular onto the second line"""
    #
    #     # print('\ndistances:')
    #     if len(self.stop_lines):
    #         line1 = self.stop_lines[0]
    #         slope1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0])
    #         y_intercept_1 = line1[0][1] - slope1 * line1[0][0]
    #
    #         for line2 in self.stop_lines[1:]:
    #             slope2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0])
    #             y_intercept_2 = line2[0][1] - slope2 * line2[0][0]
    #
    #             dist = abs(-y_intercept_1 + y_intercept_2) / math.sqrt(slope2 ** 2 + 1)
    #
    #             y_intercept_1 = y_intercept_2
    #             if dist < 500:
    #                 self.stop_lines.remove(line2)
    #             else:
    #                 # print('dist:', dist)
    #                 pass
    #
    # def detect_horizontals(self, hough_lines, center_line, right_line):
    #     stop_lines = []
    #     horizontals = []
    #     right_intersections = []
    #     center_intersections = []
    #
    #     if hough_lines is not None:
    #         for line in hough_lines:
    #             _, deg = calculate_angle_with_Ox(line)
    #             if abs(deg) < 3:
    #                 horizontals.append(line)
    #
    #     for line in horizontals:
    #         # print('bef:', line)
    #         # print('\nline:', line)
    #         # slope = (y2 - y1)/(x2 - x1)
    #         slope_center = (center_line[1][1] - center_line[0][1]) / (center_line[1][0] - center_line[0][0])
    #         slope_right = (right_line[1][1] - right_line[0][1]) / (right_line[1][0] - right_line[0][0])
    #         slope_line = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
    #
    #         # y_intercept = y - slope * x
    #         y_intercept_center = center_line[0][1] - slope_center * center_line[0][0]
    #         y_intercept_right = right_line[0][1] - slope_right * right_line[0][0]
    #         y_intercept_line = line[0][1] - slope_line * line[0][0]
    #
    #         # x_intersect = (y_intercept2 -y_intercept1) / (slope1 - slope2)
    #         x_intersect_center = (y_intercept_center - y_intercept_line) / (slope_line - slope_center)
    #         x_intersect_right = (y_intercept_right - y_intercept_line) / (slope_line - slope_right)
    #
    #         # y_intersect = slope * x_intersect + y_intercept
    #         y_intersect_center = slope_center * x_intersect_center + y_intercept_center
    #         y_intersect_right = slope_right * x_intersect_right + y_intercept_right
    #
    #         if all(not math.isnan(val) for val in
    #                [x_intersect_center, y_intersect_center, x_intersect_right, y_intersect_right]):
    #             intersect_center = (int(x_intersect_center), int(y_intersect_center))
    #             intersect_right = (int(x_intersect_right), int(y_intersect_right))
    #
    #             # print('center:', intersect_center)
    #             # print('right:', intersect_right)
    #
    #             right_intersections.append(intersect_right)
    #             center_intersections.append(intersect_center)
    #
    #             horizontal_line = (intersect_center, intersect_right)
    #             # print('aft:', horizontal_line)
    #             stop_lines.append(horizontal_line)
    #
    #     # print('stop lines:', self.stop_lines)
    #     self.stop_lines = stop_lines
    #     self.distances()
    #     return horizontals, right_intersections, center_intersections

    def extend_line(self, line: LineSegment) -> LineSegment:
        if line:
            x_bottom = int(line.compute_intersecting_x_coordinate(self.height))

            third_of_height = self.height // 2
            x_top = int(line.compute_intersecting_x_coordinate(third_of_height))

            extended_line = LineSegment(x_bottom, self.height, x_top, third_of_height)
        else:
            raise ValueError("The line segments are not valid")
        return extended_line