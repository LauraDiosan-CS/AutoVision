from objects.pipe_data import PipeData
from objects.types.line_segment import LineSegment
from objects.types.video_info import VideoInfo
import cv2
import numpy as np


def draw_sign(frame, sign):
    top_left = (int(sign.bbox[0]), int(sign.bbox[1]))
    bottom_right = (int(sign.bbox[2]), int(sign.bbox[3]))

    cv2.rectangle(frame, top_left, bottom_right, color=(0, 255, 0), thickness=3)


def draw_horizontals(frame, lines, right_int, center_int):
    for line in lines:
        cv2.line(frame, line[0], line[1], color=(255, 0, 0), thickness=2)
        cv2.circle(frame, line[0], radius=3, color=(0, 0, 255))
        cv2.circle(frame, line[1], radius=3, color=(0, 0, 255))

    for p in right_int:
        cv2.circle(frame, p, radius=3, color=(0, 0, 255))

    for p in center_int:
        cv2.circle(frame, p, radius=3, color=(0, 0, 255))


class Visualizer:
    def __init__(self, video_info: VideoInfo):
        self.width = video_info.width
        self.height = video_info.height
        self.car_position = (int(self.width / 2), self.height)
        self.center_line = None
        self.right_line = None
        self.stop_lines = None

    def draw_frame_based_on_data(self, data: PipeData):
        if data.unfiltered_frame is None:
            return data
        frame = data.unfiltered_frame.copy()
        traffic_signs = data.traffic_signs

        self.draw_car_position(frame, self.car_position)

        self.draw_command(frame, data.command)

        # drawing horizontal lines
        # data = self.filter(data)

        # draw a horizontal ine at a specific height
        cv2.line(frame, (0, self.height // 2), (frame.shape[1], self.height // 2), color=(255, 255, 255), thickness=3)

        # drawing signs info
        signs = []
        i = 0
        if len(data.traffic_signs):
            for sign in traffic_signs:
                draw_sign(frame, sign)

                signs_dict = {'Sign': sign.label, 'Confidence': sign.conf, 'Estimated Distance': sign.distance}

                signs.append(signs_dict)

            sorted_signs = sorted(signs, key=lambda x: float(x['Estimated Distance']))

            for sign in sorted_signs:
                text = ''
                for key in sign.keys():
                    text += f'{key}: {sign[key]},'

                for string in text.split(','):
                    self.put_text(frame, string, position=(0, 130 + i), color=(255, 0, 0))
                    i += 20

        # drawing lane info
        if data.road_markings is not None:
            if len(data.road_markings.stop_lines):
                stop_lines: list[LineSegment] = data.road_markings.stop_lines
                for line in stop_lines:
                    cv2.line(frame,
                             (line.upper_x, line.upper_y),
                             (line.lower_x, line.lower_y),
                             color=(0, 0, 255), thickness=3)

            center_line: LineSegment = data.road_markings.center_line
            right_line: LineSegment = data.road_markings.right_line

            if center_line and right_line:
                upper_lane_center = ((center_line.upper_x + right_line.upper_x) // 2,
                                     (center_line.upper_y + right_line.upper_y) // 2)

                self.draw_lane_endpoints(frame, center_line, right_line)
                self.draw_lanes(frame, center_line, right_line,
                                data.road_markings.center_line_virtual, data.road_markings.right_line_virtual)
                self.define_lane_area(frame, center_line, right_line)

                self.draw_correct_path(frame, self.car_position, upper_lane_center)
                self.draw_actual_path(frame, self.car_position, upper_lane_center)

                if data.heading_error is not None:
                    self.put_text(frame, f'Heading Error: {data.heading_error: .2f} degrees')
                if data.lateral_offset is not None:
                    self.put_text(frame, f'Lateral Offset: {data.lateral_offset: .2f}', color=(0,0,255), position=(0, 50))
        else:
            self.put_text(frame, "No road markings detected", position=(0, 50), color=(0, 0, 255))

        return frame

    @staticmethod
    def put_text(frame, text, position=(0, 100), font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=0.8, color=(0, 255, 255), thickness=2):
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

    @staticmethod
    def define_lane_area(frame, center_line: LineSegment, right_line: LineSegment, alpha=0.2, mask_color=(0, 204, 119)):
        lane_roi_points = np.array([(center_line.upper_x, center_line.upper_y),
                                    (right_line.upper_x, right_line.upper_y),
                                    (right_line.lower_x, right_line.lower_y),
                                    (center_line.lower_x, center_line.lower_y)], np.int32)
        lane_roi_mask = np.zeros_like(frame)
        cv2.fillPoly(lane_roi_mask, [lane_roi_points], mask_color)

        cv2.addWeighted(frame, 1, lane_roi_mask, alpha, 0, frame)

    def filter(self, data: PipeData) -> PipeData:
        image = data.frame.copy()
        gray = cv2.cvtColor(data.frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image = cv2.drawContours(image, contours, -1, (0, 255, 75), 2)
        data.add_processed_frame(image)

        return data

    @staticmethod
    def draw_lane_endpoints(frame, center_line: LineSegment, right_line: LineSegment, color_center=(255, 0, 0),
                            color_right=(0, 255, 0),
                            radius=10):
        Visualizer.draw_points(frame, [(center_line.upper_x, center_line.upper_y)], color=color_center, radius=radius)
        Visualizer.draw_points(frame, [(right_line.upper_x, right_line.upper_y)], color=color_right, radius=radius)

    @staticmethod
    def draw_correct_path(frame, car_position, upper_lane_center, color=(127, 127, 255), thickness=3):
        cv2.line(frame, car_position, upper_lane_center, color=color, thickness=thickness)

    @staticmethod
    def draw_actual_path(frame, car_position, upper_lane_center, color=(2, 135, 247), thickness=3):
        car_path_upper_limit = (car_position[0], upper_lane_center[1])
        cv2.line(frame, car_position, car_path_upper_limit, color=color, thickness=thickness)

    @staticmethod
    def draw_lanes(frame, center_line, right_line, center_line_virtual, right_line_virtual, color_center=(255, 0, 0),
                   color_right=(0, 255, 0), thickness=5):
        if center_line_virtual:
            color_center = (255, 0, 255)
        cv2.line(frame,
                 (center_line.lower_x, center_line.lower_y),
                 (center_line.upper_x, center_line.upper_y),
                 color=color_center, thickness=thickness)
        if right_line_virtual:
            color_right = (255, 0, 255)
        cv2.line(frame,
                 (right_line.lower_x, right_line.lower_y),
                 (right_line.upper_x, right_line.upper_y),
                 color=color_right, thickness=thickness)

    @staticmethod
    def draw_points(frame, endpoints, radius=10, color=(255, 0, 0), thickness=-1):
        for point in endpoints:
            cv2.circle(frame, point, radius, color, thickness=thickness)

    @staticmethod
    def draw_car_position(frame, car_position, color=(2, 135, 247), radius=20):
        Visualizer.draw_points(frame, [car_position], color=color, radius=radius)

    def draw_command(self, frame, command: str):
        self.put_text(frame, f'Command: {command}', position=(0, 20), color=(0, 255, 0))