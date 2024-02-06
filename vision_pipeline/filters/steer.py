from road_info import RoadMarkings
from vision_pipeline.filters.base_filter import BaseFilter
from video_info import VideoInfo
import cv2
import numpy as np

class Steer(BaseFilter):
    def __init__(self, video_info: VideoInfo ):
        super().__init__(video_info=video_info, return_type="steering_angle")
        self.car_position = (int(self.width / 2), self.height)
    
    def process(self, road_markings: RoadMarkings):
        center_line = road_markings.center_line
        right_line = road_markings.right_line
        print(center_line, right_line)
        if center_line and right_line:
            # defining Car Position and Lane Center
            car_position = (int(self.width / 2), self.height)

            lane_center_x = int((center_line.upper_x + right_line.upper_x) / 2)
            lane_center_y = int((center_line.upper_y + right_line.upper_y) / 2)
            self.upper_lane_center = (lane_center_x, lane_center_y)

            # calculating steering angle
            direction_vector = np.array(self.upper_lane_center) - np.array(car_position)
            steer_assist_vector = np.array([0, -1])

            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
            steer_assist_unit_vector = steer_assist_vector / np.linalg.norm(steer_assist_vector)

            dot_product = np.dot(direction_unit_vector, steer_assist_unit_vector)
            steering_angle = np.degrees(np.arccos(dot_product))

            '''if direction is negative, it means the car moves to the right of the lane center
            and needs to  move to the left to correct its position, hence the steering angle needs to be negative'''
            if direction_vector[0] < 0:
                steering_angle = -steering_angle
            print(steering_angle)

        return steering_angle
