import numpy as np
import cv2
from perception.filters.base_filter import BaseFilter
from perception.objects.line_segment import LineSegment
from perception.objects.pipe_data import PipeData
from perception.objects.road_info import RoadMarkings, RoadObject
from perception.objects.video_info import VideoInfo


def filter_lines_by_type(hough_line_segments: list[LineSegment], frame_width: int):
    horizontal_lines = []
    left_lane_lines = []
    right_lane_lines = []
    for line_segment in hough_line_segments:
        if line_segment.check_is_horizontal(20):
            horizontal_lines.append(line_segment)
        else:
            if line_segment.lower_x < frame_width / 2:
                left_lane_lines.append(line_segment)
            else:
                right_lane_lines.append(line_segment)

    return left_lane_lines, right_lane_lines, horizontal_lines


def filter_for_white_lines(color_frame, line_segments, threshold=135, num_points=50) -> tuple[
    list[LineSegment], list[LineSegment]]:

    if num_points < 1:
        raise ValueError("The number of points must be greater than 0")

    white_lines = []
    other_lines = []

    for line_segment in line_segments:
        white_points = 0

        # Generate points along the line
        discretized_points: list[tuple[int, int]] = line_segment.discretize(num_points)

        # Check each point along the line
        for x, y in discretized_points:
            # Ensure x and y are integers for indexing
            x, y = int(x), int(y)
            # Check if the point is within the frame
            if 0 <= x < color_frame.shape[1] and 0 <= y < color_frame.shape[0]:
                if all(i >= threshold for i in color_frame[y, x]):  # Check if the point is white
                    white_points += 1

        # If most of the sampled points are white, consider the line as on a white part
        if white_points / num_points > 0.5:
            white_lines.append(line_segment)
        else:
            other_lines.append(line_segment)

    return white_lines, other_lines


def draw_lines(frame, line_segments_with_colors):
    for color, line_segments_with_thickness in line_segments_with_colors.items():
        line_segments, thickness = line_segments_with_thickness
        if line_segments:
            for line_segment in line_segments:
                cv2.line(frame, line_segment.lower_point, line_segment.upper_point, color, thickness)
    return frame


def visualize_hough_lines(data: PipeData, lane_white_horizontal_lines, left_line_segment, other_horizontal_lines,
                          other_left_lane_lines, other_right_lane_lines, right_line_segment, white_horizontal_lines,
                          white_horizontals_outside_of_lane, white_left_lane_lines, white_right_lane_lines):
    color_mapping_bgr = {
        "cyan": (255, 255, 0),
        "dark_blue": (175, 0, 0),
        "light_blue": (173, 216, 230),
        "violet": (238, 130, 238),
        "lime": (0, 255, 0),
        "dark_green": (0, 100, 0),
        "light_green": (144, 238, 144),
        "yellow": (255, 255, 0),
        "red": (0, 0, 255),
        "pink": (255, 192, 203),
        "orange": (0, 165, 255)
    }
    frame = draw_lines(data.frame.copy(), line_segments_with_colors={
        color_mapping_bgr["dark_blue"]: (other_left_lane_lines, 1),
        color_mapping_bgr["dark_green"]: (other_right_lane_lines, 1),
        color_mapping_bgr["orange"]: (other_horizontal_lines, 1)
    })
    data.add_processed_frame(frame)
    frame = draw_lines(data.frame.copy(), line_segments_with_colors={
        color_mapping_bgr["cyan"]: (white_left_lane_lines, 2),
        color_mapping_bgr["light_green"]: (white_right_lane_lines, 2),
        color_mapping_bgr["pink"]: (white_horizontal_lines, 1)
    })
    data.add_processed_frame(frame)
    frame = draw_lines(data.frame.copy(), line_segments_with_colors={
        color_mapping_bgr["cyan"]: ([left_line_segment], 5) if left_line_segment else ([], 5),
        color_mapping_bgr["lime"]: ([right_line_segment], 5) if right_line_segment else ([], 5),
        color_mapping_bgr["red"]: (lane_white_horizontal_lines, 5),
        color_mapping_bgr["violet"]: (white_horizontals_outside_of_lane, 2),
    })
    data.add_processed_frame(frame)


class LaneDetectFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool):
        super().__init__(video_info=video_info, visualize=visualize)

        self.max_lane_height = self.video_height // 2
        # Pixels per cm =   Width / Width in cm (or Height / Height in cm)
        self.lane_width_in_cm = 35
        self.camera_visible_width_in_cm = 51
        self.lane_width_in_pixels = int((self.video_width / self.camera_visible_width_in_cm) * self.lane_width_in_cm)

    def process(self, data: PipeData) -> PipeData:
        hough_lines = self.apply_houghLines(data.frame)

        if hough_lines is None:
            return super().process(data)

        hough_line_segments = [] # list of LineSegment objects with (x1, y1) being the upper point (lower in height)
        for i in range(len(hough_lines)):
            x1, y1, x2, y2 = hough_lines[i][0]
            if y1 < y2:  # point 1 is the upper point
                hough_line_segments.append(LineSegment(x2, y2, x1, y1))
            else:
                hough_line_segments.append(LineSegment(x1, y1, x2, y2))

        white_line_segments, other_line_segments = filter_for_white_lines(data.unfiltered_frame, hough_line_segments)

        white_left_lines, white_right_lines, white_horizontal_lines = filter_lines_by_type(
            white_line_segments,
            self.video_width
        )
        other_left_lines, other_right_lines, other_horizontal_lines = filter_lines_by_type(
            other_line_segments,
            self.video_width
        )

        left_line_segment = None
        if len(white_left_lines) > 0:
            left_line_segment = max(white_left_lines, key=lambda l: l.compute_vertical_distance())
            left_line_segment = self.extend_line(left_line_segment)

        right_line_segment = None
        if len(white_right_lines) > 0:
            right_line_segment = max(white_right_lines, key=lambda l: l.compute_euclidean_distance())
            right_line_segment = self.extend_line(right_line_segment)

        left_line_virtual = False
        if right_line_segment is not None and left_line_segment is None:
            left_line_segment = self.compute_virtual_left_lane(self.camera_visible_width_in_cm, self.lane_width_in_pixels,
                                                               right_line_segment, threshold_cm=5)
            left_line_virtual = True

        right_line_virtual = False
        if left_line_segment is not None and right_line_segment is None:
            right_line_segment = self.compute_virtual_right_lane(self.camera_visible_width_in_cm, self.lane_width_in_pixels,
                                                                 left_line_segment, threshold_cm=12)
            right_line_virtual = True

        if left_line_segment and right_line_segment:
            half_lane_distance = (right_line_segment.lower_x - left_line_segment.lower_x) / 2
            dist_to_left_lane = self.video_width / 2 - left_line_segment.lower_x
            data.lateral_offset = (dist_to_left_lane - half_lane_distance) / (half_lane_distance + 0.0001)  # avoid division by zero

        lane_white_horizontal_lines, white_horizontals_outside_of_lane = self.filter_horizontals_based_on_lane(
            white_horizontal_lines,
            left_line_segment,
            right_line_segment)

        data.road_markings = RoadMarkings(left_line=None,
                                          center_line=left_line_segment,
                                          right_line=right_line_segment,
                                          stop_lines=lane_white_horizontal_lines,
                                          center_line_virtual=left_line_virtual,
                                          right_line_virtual=right_line_virtual,
                                          )

        horiz_line_objects = [RoadObject(bbox=[[horiz_line_segment.lower_x, horiz_line_segment.lower_y],
                                               [horiz_line_segment.upper_x, horiz_line_segment.upper_y]],
                                         label="horiz_line",
                                         conf=1,
                                         distance=0) for horiz_line_segment in lane_white_horizontal_lines]

        data.horizontal_lines = horiz_line_objects

        if left_line_segment and right_line_segment:
            data.frame = cv2.cvtColor(data.frame, cv2.COLOR_GRAY2BGR)
        if self.visualize:
            visualize_hough_lines(data, lane_white_horizontal_lines, left_line_segment, other_horizontal_lines,
                                  other_left_lines, other_right_lines, right_line_segment,
                                  white_horizontal_lines, white_horizontals_outside_of_lane, white_left_lines,
                                  white_right_lines)
            return data  # skip visualization from base filter

        return super().process(data)

    def compute_virtual_right_lane(self, camera_width_in_cm, lane_width_in_pixels, left_line_segment, threshold_cm):
        right_lower_y = left_line_segment.lower_y
        right_upper_y = left_line_segment.upper_y

        x_distance_between_left_line_endings = abs(left_line_segment.upper_x - left_line_segment.lower_x)

        dist_between_upper_points = lane_width_in_pixels - 2 * x_distance_between_left_line_endings
        th_px = int((self.video_width / camera_width_in_cm) * threshold_cm)

        # If the distance between the upper points is less than the threshold
        if dist_between_upper_points < th_px:  # we are turning left
            right_lower_x = left_line_segment.lower_x + lane_width_in_pixels
            right_upper_x = left_line_segment.upper_x + lane_width_in_pixels
        else:  # we are going straight
            right_lower_x = left_line_segment.lower_x + lane_width_in_pixels
            right_upper_x = right_lower_x - x_distance_between_left_line_endings

        return LineSegment(right_lower_x, right_lower_y, right_upper_x, right_upper_y)

    def compute_virtual_left_lane(self, camera_width_in_cm, lane_width_in_pixels, right_line_segment, threshold_cm):
        left_lower_y = right_line_segment.lower_y
        left_upper_y = right_line_segment.upper_y

        x_distance_between_right_line_endings = abs(right_line_segment.upper_x - right_line_segment.lower_x)

        dist_between_upper_points = lane_width_in_pixels - 2 * x_distance_between_right_line_endings
        th_px = int((self.video_width / camera_width_in_cm) * threshold_cm)

        # If the distance between the upper points is less than the threshold
        if dist_between_upper_points < th_px:
            # we are turning left
            left_upper_x = right_line_segment.upper_x - lane_width_in_pixels
            left_lower_x = right_line_segment.lower_x - lane_width_in_pixels
        else:  # we are going straight
            left_lower_x = right_line_segment.lower_x - lane_width_in_pixels
            left_upper_x = left_lower_x + x_distance_between_right_line_endings

        return LineSegment(left_lower_x, left_lower_y, left_upper_x, left_upper_y)

    @staticmethod
    def apply_houghLines(frame, rho=1, theta=np.pi / 180, threshold=50, min_line_length=300, max_line_gap=200):
        return cv2.HoughLinesP(frame, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                               maxLineGap=max_line_gap)

    def filter_horizontals_based_on_lane(self, horizontal_line_segments: list[LineSegment], left_line: LineSegment,
                                         right_line: LineSegment, threshold=0.4) -> tuple[list[LineSegment], list[LineSegment]]:
        filtered_horizontals = []
        horizontals_outside_of_lane = []
        for horiz_line_segment in horizontal_line_segments:
            if left_line and right_line:
                if horiz_line_segment.lower_y < 600:  # if the line is above the lane
                    horizontals_outside_of_lane.append(horiz_line_segment)
                    continue

                if horiz_line_segment.upper_x < horiz_line_segment.lower_x:
                    left_endpoint = horiz_line_segment.upper_point
                    right_endpoint = horiz_line_segment.lower_point
                else:
                    left_endpoint = horiz_line_segment.lower_point
                    right_endpoint = horiz_line_segment.upper_point

                left_intersect_point = left_line.compute_interesting_point(horiz_line_segment)
                right_intersect_point = right_line.compute_interesting_point(horiz_line_segment)

                if left_intersect_point is None or right_intersect_point is None:
                    raise ValueError("Somehow lane lines and horizontals are parallel")

                left_distance = euclidean_distance(left_intersect_point, left_endpoint)
                right_distance = euclidean_distance(right_intersect_point, right_endpoint)

                horiz_line_segment_length = horiz_line_segment.compute_euclidean_distance()
                value = (left_distance + right_distance) / (horiz_line_segment_length + 0.0001)  # avoid division by 0

                if value < threshold:
                    filtered_horizontals.append(horiz_line_segment)
                else:
                    horizontals_outside_of_lane.append(horiz_line_segment)
        return filtered_horizontals, horizontals_outside_of_lane

    def extend_line(self, line: LineSegment) -> LineSegment:
        x_bottom = int(line.compute_intersecting_x_coordinate(self.video_height))

        x_top = int(line.compute_intersecting_x_coordinate(self.max_lane_height))

        return LineSegment(x_bottom, self.video_height, x_top, self.max_lane_height)

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)