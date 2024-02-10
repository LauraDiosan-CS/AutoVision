from objects.pipe_data import PipeData
from objects.video_info import VideoInfo
from vision_pipeline.filters.base_filter import BaseFilter
import numpy as np


class HeadingErrorFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo):
        super().__init__(video_info=video_info)
        self.car_position = (int(self.width / 2), self.height)

    def process(self, data: PipeData) -> PipeData:
        if data.road_markings is None:
            return data
        center_line = data.road_markings.center_line
        right_line = data.road_markings.right_line
        print(center_line, right_line)
        if center_line and right_line:
            upper_lane_center = ((center_line.upper_point.x + right_line.upper_point.x) // 2,
                                 (center_line.upper_point.y + right_line.upper_point.y) // 2)

            # calculating steering angle
            direction_vector = np.array(upper_lane_center) - np.array(self.car_position)
            steer_assist_vector = np.array([0, -1])

            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
            steer_assist_unit_vector = steer_assist_vector / np.linalg.norm(steer_assist_vector)

            dot_product = np.dot(direction_unit_vector, steer_assist_unit_vector)
            steering_angle = np.degrees(np.arccos(dot_product))

            '''if direction is negative, it means the car moves to the right of the lane center
            and needs to  move to the left to correct its position, hence the steering angle needs to be negative'''
            if direction_vector[0] < 0:
                steering_angle = -steering_angle
            data.steering_angle = steering_angle

        return data