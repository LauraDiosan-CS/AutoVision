from vision_pipeline.filters.base_filter import BaseFilter
from video_info import VideoInfo
import cv2
import numpy as np

class DrawFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo ):
        super().__init__(video_info=video_info, return_type="img")
        self.car_position = (int(self.width / 2), self.height)
        self.center_line = None
        self.right_line = None

    def process(self, frame, road_markings, steering_angle):
        self.center_line = road_markings.center_line
        self.right_line = road_markings.right_line

        lane_center_x = int((self.center_line.upper_x + self.right_line.upper_x) / 2)
        lane_center_y = int((self.center_line.upper_y + self.right_line.upper_y) / 2)
        self.upper_lane_center = (lane_center_x, lane_center_y)

        self.draw_lanes(frame)
        self.define_lane_area(frame)
        self.put_text_steering_angle(frame, steering_angle)
        self.draw_correct_path(frame=frame)
        self.draw_actual_path(frame=frame)
        self.draw_car_position(frame)
        self.draw_lane_endpoints(frame)
        return frame

    def put_text_steering_angle(self, frame, steering_angle):
        cv2.putText(frame, f'Steering angle: {steering_angle:.2f}', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 6)

    def define_lane_area(self, frame):
        if self.center_line and self.right_line:
            roi_points = np.array([
                (self.center_line.upper_x, self.center_line.upper_y), 
                (self.center_line.lower_x, self.center_line.lower_y),
                (self.right_line.lower_x, self.right_line.lower_y),
                (self.right_line.upper_x, self.right_line.upper_y)],
                dtype=np.int32)
            
            # mask for ROI
            roi_mask = np.zeros_like(frame)
            cv2.fillPoly(roi_mask, [roi_points], (0, 204, 119))

            # combine frame with roi masking, adding a transparency factor
            alpha = 0.2
            cv2.addWeighted(frame, 1, roi_mask, alpha, 0, frame)

    def draw_lane_endpoints(self, frame):
        print('center line:', type(int(self.center_line[0])))
        print('right line:', type(int(self.right_line[0])))
        self.draw_points(frame, [(int(self.center_line[0]), int(self.center_line[1]))],color=(255,0,0), radius=10)
        self.draw_points(frame, [(int(self.right_line[0]), int(self.right_line[1]))], color=(0, 255, 0), radius = 10)

    def draw_correct_path(self, frame):
        if self.center_line and self.right_line:
            if self.car_position and self.upper_lane_center:
                print('correct:', self.car_position, self.upper_lane_center)
                cv2.line(frame, self.car_position, self.upper_lane_center, color=(0, 255, 0), thickness=3)

    def draw_actual_path(self, frame):
        if self.center_line and self.right_line:
            if self.car_position and self.upper_lane_center:
                print('actual:', self.car_position, self.upper_lane_center)
                car_path_upper_limit = (self.car_position[0], self.upper_lane_center[1])
                cv2.line(frame, self.car_position, car_path_upper_limit, color=(2, 135, 247), thickness=3)

    def draw_lanes(self, frame):
        if self.center_line and self.right_line:
            cv2.line(frame, (self.center_line.upper_x, self.center_line.upper_y), 
                (self.center_line.lower_x, self.center_line.lower_y),  color=(0, 255, 0), thickness=5)
            cv2.line(frame, (self.right_line.upper_x, self.right_line.upper_y), 
                (self.right_line.lower_x, self.right_line.lower_y), color=(0, 255, 0), thickness=5)

    def draw_points(self, frame, endpoints, color=(255, 0, 0), radius=10):
        if endpoints:
            for point in endpoints:
                cv2.circle(frame, point, radius, color, thickness=-1)

    def draw_car_position(self, frame):
        self.draw_points(frame,[self.car_position], color=(2, 135, 247),radius=20)

    def draw_lane_center(self, frame):
        self.draw_points(frame,[self.upper_lane_center], radius=10)