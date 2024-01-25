import numpy as np
import math
import cv2


class LaneDetect():

    def __init__(self):
        self.frame = None
        self.height = None
        self.width = None
        self.roi = None
        self.trapezoid_vertices = None
        self.left_lane = []
        self.right_lane = []
        self.car_position = None
        self.lane_center = None
        self.steering = None

    def set_frame(self, frame):
        if frame is not None:
            self.frame = frame
            self.height, self.width, _ = frame.shape

    def send_frame(self):
        return self.frame

    def get_steering_angle(self):
        return self.steering

    def process(self, frame):
        self.set_frame(frame)

        # frame preprocessing
        canny_edges = self.cannyEdge()
        masked_edges = self.define_roi(canny_edges)
        hough_lines = self.houghLines(masked_edges)

        # lane detection
        self.detect_max_lanes(hough_lines)
        self.extend_lanes()
        self.draw_lanes()
        self.define_lane_area()

        # calculate steering
        self.steering_angle()

        # visualize processed frame
        self.visualize()

        # send results
        return self.send_frame()

    # -----------------------------------------------
    # Processing Methods    

    def cannyEdge(self, low_threshold=150, high_threshold=250):
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, low_threshold, high_threshold)
        return edges

    def define_roi(self, canny_edges):
        if self.roi is None:
            height, width = canny_edges.shape
            self.trapezoid_vertices = np.array([[(width * 0 - 100, height), (width * 0.2, height * 0.6),
                                                 (width * 0.8, height * 0.6), (width + 100, height)]], dtype=np.int32)
            mask = np.zeros_like(canny_edges)
            cv2.fillPoly(mask, self.trapezoid_vertices, 255)
            self.roi = mask
        masked_edges = cv2.bitwise_and(canny_edges, self.roi)
        return masked_edges

    def houghLines(self, masked_edges, rho=1, theta=np.pi / 180, threshold=50, min_line_length=100, max_line_gap=550):
        return cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
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

    def calculate_angle_with_Ox(self, line):
        for x1, y1, x2, y2 in line:
            delta_x = x2 - x1
            delta_y = y2 - y1
            angle_rad = math.atan2(delta_y, delta_x)
            angle_deg = math.degrees(angle_rad)
        return angle_rad, angle_deg

    def detect_max_lanes(self, hough_lines):

        max_left_lane, max_right_lane = self.max_lane_lengths(hough_lines)
        # resetting lanes
        self.left_lane, self.right_lane = [], []

        if hough_lines is not None:
            for line in hough_lines:
                is_horizontal = False
                for x1, y1, x2, y2 in line:
                    # checking  if the line is approximately vertical
                    _, deg = self.calculate_angle_with_Ox(line)

                    if abs(deg) < 5:
                        is_horizontal = True

                    coords = line[0]
                    if not ((abs(y1 - y2) < 50) or (abs(x1 - x2) < 100)):

                        if x2 < self.width / 2 and abs(
                                y1 - y2) == max_left_lane and not is_horizontal and not self.left_lane:
                            for i in range(2):
                                self.left_lane.append((coords[2 * i], coords[2 * i + 1]))

                        elif x2 > self.width / 2 and abs(
                                y1 - y2) == max_right_lane and not is_horizontal and not self.right_lane:
                            for i in range(2):
                                self.right_lane.append((coords[2 * i], coords[2 * i + 1]))

    def lane_distance(self, lane_endpoints):
        x1 = lane_endpoints[0][0]
        y1 = lane_endpoints[0][1]
        x2 = lane_endpoints[1][0]
        y2 = lane_endpoints[1][1]

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def find_intersection_x_coordinate(self, slope, y_intercept, y_horizontal_line):
        return int((y_horizontal_line - y_intercept) / slope)

    def extend_lanes(self):
        # left and right lanes extended down until lower limit
        ext_lower_left = []
        ext_lower_right = []
        # left and right lanes extended up untill upper limit
        ext_upper_left = []
        ext_upper_right = []

        if self.left_lane and self.right_lane:
            # 1. extending both lanes downuntil the bottom of frame
            left_slope = (self.left_lane[1][1] - self.left_lane[0][1]) / (self.left_lane[1][0] - self.left_lane[0][0])
            right_slope = (self.right_lane[1][1] - self.right_lane[0][1]) / (
                        self.right_lane[1][0] - self.right_lane[0][0])

            y_intercept_left = self.left_lane[0][1] - left_slope * self.left_lane[0][0]
            y_intercept_right = self.right_lane[0][1] - right_slope * self.right_lane[0][0]

            x_left = (self.height - y_intercept_left) / left_slope
            bottom_left = (int(x_left), self.height)

            x_right = (self.height - y_intercept_right) / right_slope
            bottom_right = (int(x_right), self.height)

            ext_lower_left.append(self.left_lane[1])
            ext_lower_left.append(bottom_left)

            ext_lower_right.append(self.right_lane[0])
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

        self.left_lane = ext_upper_left
        self.right_lane = ext_upper_right

    def steering_angle(self):
        if self.left_lane and self.right_lane:
            # defining Car Position and Lane Center
            self.car_position = (int(self.width / 2), self.height)

            x_lane_center = int((self.left_lane[0][0] + self.right_lane[0][0]) / 2)
            y_lane_center = int((self.left_lane[0][1] + self.right_lane[0][1]) / 2)
            self.lane_center = (x_lane_center, y_lane_center)

            # calculating steering angle
            direction_vector = np.array(self.lane_center) - np.array(self.car_position)
            steer_assist_vector = np.array([0, -1])

            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
            steer_assist_unit_vector = steer_assist_vector / np.linalg.norm(steer_assist_vector)

            dot_product = np.dot(direction_unit_vector, steer_assist_unit_vector)
            steering_angle = np.degrees(np.arccos(dot_product))

            '''if direction is negative, it means the car moves to the right of the lane center
            and needs to  move to the left to correct its position, hence the steering angle needs to be negative'''
            if direction_vector[0] < 0:
                steering_angle = -steering_angle

            self.steering = steering_angle
            cv2.putText(self.frame, f'Steering angle: {steering_angle:.2f}', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 255), 6)

    # -----------------------------------------------
    # Visualization Methods

    def visualize(self):
        self.draw_correct_path()
        self.draw_actual_path()
        self.draw_car_position()

    def define_lane_area(self):
        if self.left_lane and self.right_lane:
            roi_points = np.array([self.left_lane[0], self.left_lane[1], self.right_lane[1], self.right_lane[0]],
                                  dtype=np.int32)

            # mask for ROI
            roi_mask = np.zeros_like(self.frame)
            cv2.fillPoly(roi_mask, [roi_points], (0, 204, 119))

            # combine frame with roi masking, adding a transparency factor
            alpha = 0.2
            cv2.addWeighted(self.frame, 1, roi_mask, alpha, 0, self.frame)

    def draw_correct_path(self):
        if self.left_lane and self.right_lane:
            if self.car_position and self.lane_center:
                print('correct:', self.car_position, self.lane_center)
                cv2.line(self.frame, self.car_position, self.lane_center, color=(0, 255, 0), thickness=3)

    def draw_actual_path(self):
        if self.left_lane and self.right_lane:
            if self.car_position and self.lane_center:
                print('actual:', self.car_position, self.lane_center)
                car_path_upper_limit = (self.car_position[0], self.lane_center[1])
                cv2.line(self.frame, self.car_position, car_path_upper_limit, color=(255, 255, 255), thickness=3)

    def draw_lanes(self):
        if self.left_lane and self.right_lane:
            cv2.line(self.frame, self.left_lane[0], self.left_lane[1], color=(0, 255, 0), thickness=5)
            cv2.line(self.frame, self.right_lane[0], self.right_lane[1], color=(0, 255, 0), thickness=5)

    def draw_endpoints(self, endpoints, color=(0, 255, 0), radius=10):
        if endpoints:
            for point in endpoints:
                cv2.circle(self.frame, point, radius, color, thickness=-1)

    def draw_car_position(self):
        self.draw_endpoints([self.car_position], radius=20)

    def draw_lane_center(self):
        self.draw_endpoints([self.lane_center], radius=10)

    def draw_lane_endpoints(self):
        self.draw_endpoints([self.left_lane[0], self.right_lane[0]], radius=10)