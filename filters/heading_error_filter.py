from objects.pipe_data import PipeData
from objects.types.line_segment import LineSegment
from objects.types.video_info import VideoInfo
from filters.base_filter import BaseFilter
import numpy as np


class HeadingErrorFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo, visualize: bool):
        super().__init__(video_info=video_info, visualize=visualize)
        self.car_position = (int(self.width / 2), self.height)

    def process(self, data: PipeData) -> PipeData:
        if data.road_markings is None:
            return data
        center_line: LineSegment = data.road_markings.center_line
        right_line: LineSegment = data.road_markings.right_line

        if center_line and right_line:
            lane_distance = right_line.lower_x - center_line.lower_x
            half_lane_distance = lane_distance / 2
            dist_to_left_lane = self.width / 2 - center_line.lower_x
            data.lateral_offset = (dist_to_left_lane - half_lane_distance) / half_lane_distance

            # distance_to_right_lane = right_line.lower_x - self.width // 2
            # distance_to_left_lane = self.width // 2 - center_line.lower_x
            #
            # total_distance = distance_to_right_lane + distance_to_left_lane
            #
            # data.lateral_offset = -(2 * distance_to_right_lane / total_distance) + 1
            #

            upper_lane_center = ((center_line.upper_x + right_line.upper_x) // 2,
                                 (center_line.upper_y + right_line.upper_y) // 2)

            # calculating heading_error
            direction_vector = np.array(upper_lane_center) - np.array(self.car_position)
            steer_assist_vector = np.array([0, -1])

            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
            steer_assist_unit_vector = steer_assist_vector / np.linalg.norm(steer_assist_vector)

            dot_product = np.dot(direction_unit_vector, steer_assist_unit_vector)
            heading_error = np.degrees(np.arccos(dot_product))

            '''if direction is negative, it means the car moves to the right of the lane center
            and needs to  move to the left to correct its position, hence the steering angle needs to be negative'''
            if direction_vector[0] < 0:
                heading_error = -heading_error
            data.heading_error = -heading_error

        return super().process(data)