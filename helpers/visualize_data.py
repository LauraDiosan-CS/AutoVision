import cv2
import numpy as np
from objects.pipe_data import PipeData
from objects.types.line_segment import LineSegment
from objects.types.road_info import RoadObject
from objects.types.video_info import VideoInfo


def draw_bbox(frame, bbox, color):
    top_left = (int(bbox[0]), int(bbox[1]))
    bottom_right = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(frame, top_left, bottom_right, color=color, thickness=3)


def visualize_road_objects(frame, road_objects: list[RoadObject], color=(255, 255, 255), initial_position=(0, 130), font_scale=0.8):
    # Sort the traffic signs by distance
    sorted_objects = sorted(road_objects, key=lambda obj: obj.distance)

    # Display each sign's information on the frame
    for obj in sorted_objects:
        draw_bbox(frame, obj.bbox, color)
        obj_info = ', '.join([f'{key}: {value}' for key, value in obj._asdict().items() if key != 'bbox'])
        cv2.putText(frame, obj_info, initial_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        initial_position = (initial_position[0], initial_position[1] + 20)


def put_text(frame, text, position=(0, 100), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, color=(0, 255, 255), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, color, thickness)


def define_lane_area(frame, center_line: LineSegment, right_line: LineSegment, alpha=0.2, mask_color=(0, 204, 119)):
    lane_roi_points = np.array([(center_line.upper_x, center_line.upper_y),
                                (right_line.upper_x, right_line.upper_y),
                                (right_line.lower_x, right_line.lower_y),
                                (center_line.lower_x, center_line.lower_y)], np.int32)
    lane_roi_mask = np.zeros_like(frame)
    cv2.fillPoly(lane_roi_mask, [lane_roi_points], mask_color)
    cv2.addWeighted(frame, 1, lane_roi_mask, alpha, 0, frame)


def draw_lane_endpoints(frame, center_line: LineSegment, right_line: LineSegment, color_center=(255, 0, 0), color_right=(0, 255, 0), radius=10):
    draw_points(frame, [(center_line.upper_x, center_line.upper_y)], color=color_center, radius=radius)
    draw_points(frame, [(right_line.upper_x, right_line.upper_y)], color=color_right, radius=radius)


def draw_correct_path(frame, car_position, upper_lane_center, color=(127, 127, 255), thickness=3):
    cv2.line(frame, car_position, upper_lane_center, color=color, thickness=thickness)


def draw_actual_path(frame, car_position, upper_lane_center, color=(2, 135, 247), thickness=3):
    car_path_upper_limit = (car_position[0], upper_lane_center[1])
    cv2.line(frame, car_position, car_path_upper_limit, color=color, thickness=thickness)


def draw_lanes(frame, center_line, right_line, center_line_virtual, right_line_virtual, color_center=(255, 0, 0), color_right=(0, 255, 0), thickness=5):
    if center_line_virtual:
        color_center = (255, 0, 255)
    cv2.line(frame, (center_line.lower_x, center_line.lower_y), (center_line.upper_x, center_line.upper_y), color=color_center, thickness=thickness)
    if right_line_virtual:
        color_right = (255, 0, 255)
    cv2.line(frame, (right_line.lower_x, right_line.lower_y), (right_line.upper_x, right_line.upper_y), color=color_right, thickness=thickness)


def draw_points(frame, endpoints, radius=10, color=(255, 0, 0), thickness=-1):
    for point in endpoints:
        cv2.circle(frame, point, radius, color, thickness=thickness)


def draw_car_position(frame, car_position, color=(2, 135, 247), radius=20):
    draw_points(frame, [car_position], color=color, radius=radius)


def display_behaviour(frame, behaviour: str):
    put_text(frame, f'Behaviour: {behaviour}', position=(0, 20), color=(0, 0, 255))


def visualize_data(video_info: VideoInfo, data: PipeData) -> PipeData:
    car_position = (int(video_info.width / 2), video_info.height)

    if data.unfiltered_frame is None:
        return data

    frame = data.unfiltered_frame.copy()
    draw_car_position(frame, car_position)
    display_behaviour(frame, data.behaviour)

    cv2.line(frame, (0, 600), (frame.shape[1], 600), color=(255, 255, 255), thickness=3)

    visualize_road_objects(frame, data.traffic_signs, initial_position=(0, 130), color=(0, 0, 255), font_scale=0.6)
    visualize_road_objects(frame, data.traffic_lights, initial_position=(0, 250), color=(0, 255, 0), font_scale=0.6)
    visualize_road_objects(frame, data.pedestrians, initial_position=(0, 400), color=(255, 0, 0), font_scale=0.6)

    if data.road_markings is not None:
        if len(data.road_markings.stop_lines):
            stop_lines: list[LineSegment] = data.road_markings.stop_lines
            for line in stop_lines:
                cv2.line(frame, (line.upper_x, line.upper_y), (line.lower_x, line.lower_y), color=(0, 0, 255), thickness=3)

        center_line: LineSegment = data.road_markings.center_line
        right_line: LineSegment = data.road_markings.right_line

        if center_line and right_line:
            upper_lane_center = ((center_line.upper_x + right_line.upper_x) // 2, (center_line.upper_y + right_line.upper_y) // 2)

            draw_lane_endpoints(frame, center_line, right_line)
            draw_lanes(frame, center_line, right_line, data.road_markings.center_line_virtual, data.road_markings.right_line_virtual)
            define_lane_area(frame, center_line, right_line)

            draw_correct_path(frame, car_position, upper_lane_center)
            draw_actual_path(frame, car_position, upper_lane_center)

            if data.heading_error is not None:
                put_text(frame, f'Heading Error: {data.heading_error: .2f} degrees')
            if data.lateral_offset is not None:
                put_text(frame, f'Lateral Offset: {data.lateral_offset: .2f}', color=(0, 0, 255), position=(0, 50))
    else:
        put_text(frame, "No road markings detected", position=(0, 50), color=(0, 0, 255))

    return frame