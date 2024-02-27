from copy import deepcopy

from objects.pipe_data import PipeData
from objects.types.road_info import Line
from objects.types.video_info import VideoInfo
from filters.base_filter import BaseFilter
import cv2
import numpy as np


class DrawFilter(BaseFilter):
    def __init__(self, video_info: VideoInfo):
        super().__init__(video_info=video_info)
        self.car_position = (int(self.width / 2), self.height)
        self.center_line = None
        self.right_line = None
              
    def process(self, data: PipeData) -> PipeData:
        if data.unfiltered_frame is None:
            return data
        frame = data.unfiltered_frame
        traffic_signs = data.traffic_signs

        self.draw_car_position(frame, self.car_position)
        signs = []

        # drawing signs info
        i = 0
        if len(data.traffic_signs):
            for sign in traffic_signs:
                self.draw_sign(frame, sign)

                signs_dict = {'Sign':'', 'Confidence':'', 'Estimated Distance': ''}
                signs_dict['Sign'] = sign.label
                signs_dict['Confidence'] = sign.conf 
                signs_dict['Estimated Distance'] = 0
                
                signs.append(signs_dict)
            
            sorted_signs = sorted(signs, key = lambda x: float(x['Estimated Distance']))
            
            for sign in sorted_signs:
                text = ''
                for key in sign.keys():
                    text += f'{key}: {sign[key]},'
                
                for string in text.split(','):
                    self.put_text(frame, string, position=(0, 100 + i), color=(255, 0, 0))
                    i += 20

        # drawing lane info
        if data.road_markings is not None:
            center_line = data.road_markings.center_line
            right_line = data.road_markings.right_line

            upper_lane_center = ((center_line.upper_point.x + right_line.upper_point.x) // 2,
                                 (center_line.upper_point.y + right_line.upper_point.y) // 2)

            self.draw_lane_endpoints(frame, center_line, right_line)
            self.draw_lanes(frame, center_line, right_line)
            self.define_lane_area(frame, center_line, right_line)

            self.draw_correct_path(frame, self.car_position, upper_lane_center)
            self.draw_actual_path(frame, self.car_position, upper_lane_center)

            if data.heading_error is not None:
                self.put_text(frame, f'Heading Error: {data.heading_error:.2f} degrees')
        else:
            self.put_text(frame, "No road markings detected", position=(0, 50), color=(0, 0, 255))

        data.frame = frame
        data.processed_frames.append(deepcopy(data.frame))
        return data

    @staticmethod
    def put_text(frame, text, position=(0, 100), font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=0.8, color=(0, 255, 255), thickness=2):
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

    @staticmethod
    def define_lane_area(frame, center_line: Line, right_line: Line, alpha=0.2, mask_color=(0, 204, 119)):
        lane_roi_points = np.array([
            center_line.upper_point,
            center_line.lower_point,
            right_line.lower_point,
            right_line.upper_point],
            dtype=np.int32)

        lane_roi_mask = np.zeros_like(frame)
        cv2.fillPoly(lane_roi_mask, [lane_roi_points], mask_color)

        cv2.addWeighted(frame, 1, lane_roi_mask, alpha, 0, frame)

    def draw_sign(self, frame, sign):
        top_left = (int(sign.bbox[0]), int(sign.bbox[1]))
        bottom_right = (int(sign.bbox[2]), int(sign.bbox[3]))

        cv2.rectangle(frame, top_left, bottom_right, color=(0, 255, 0), thickness = 3)

    @staticmethod
    def draw_lane_endpoints(frame, center_line: Line, right_line: Line, color_center=(255, 0, 0),
                            color_right=(0, 255, 0),
                            radius=10):
        DrawFilter.draw_points(frame, [center_line.upper_point], color=color_center, radius=radius)
        DrawFilter.draw_points(frame, [right_line.upper_point], color=color_right, radius=radius)

    @staticmethod
    def draw_correct_path(frame, car_position, upper_lane_center, color=(0, 255, 0), thickness=3):
        cv2.line(frame, car_position, upper_lane_center, color=color, thickness=thickness)

    @staticmethod
    def draw_actual_path(frame, car_position, upper_lane_center, color=(2, 135, 247), thickness=3):
        car_path_upper_limit = (car_position[0], upper_lane_center[1])
        cv2.line(frame, car_position, car_path_upper_limit, color=color, thickness=thickness)

    @staticmethod
    def draw_lanes(frame, center_line, right_line, color=(0, 255, 0), thickness=5):
        cv2.line(frame, center_line.upper_point, center_line.lower_point, color=color, thickness=thickness)
        cv2.line(frame, right_line.upper_point, right_line.lower_point, color=color, thickness=thickness)

    @staticmethod
    def draw_points(frame, endpoints, radius=10, color=(255, 0, 0), thickness=-1):
        for point in endpoints:
            cv2.circle(frame, point, radius, color, thickness=thickness)

    @staticmethod
    def draw_car_position(frame, car_position, color=(2, 135, 247), radius=20):
        DrawFilter.draw_points(frame, [car_position], color=color, radius=radius)