import cv2
import numpy as np
from perception.objects.line_segment import LineSegment
from perception.objects.pipe_data import PipeData
from perception.objects.road_info import RoadObject
from perception.objects.video_info import VideoInfo


def draw_bbox(frame, bbox, color):
    top_left = (int(bbox[0]), int(bbox[1]))
    bottom_right = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(frame, top_left, bottom_right, color=color, thickness=3)


def put_text(frame, text, position, font_scale, text_color=(0, 255, 255), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def put_text_with_background(frame, text, position, font_scale, thickness,text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # Draw filled rectangle for background
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, cv2.FILLED)
    # Put text over the rectangle
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

def visualize_road_objects(frame, road_objects: list[RoadObject], color, font_scale, thickness):
    if road_objects is None or len(road_objects) == 0:
        return

    # Sort the road objects by distance
    sorted_objects = sorted(road_objects, key=lambda obj: obj.distance)

    # Define padding between text and bounding box
    padding = int(15 * font_scale)  # 10 pixels scaled

    # Display each object's information above its bounding box
    for obj in sorted_objects:
        draw_bbox(frame, obj.bbox, color)

        # Prepare separate lines for label and "confidence distance"
        label_text = f"{obj.label}"
        conf_distance_text = f"{obj.conf * 100:.0f}%"
        if obj.distance != float("inf"):
            conf_distance_text += f" {obj.distance:.2f}m"

        # Calculate text sizes
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (conf_width, conf_height), _ = cv2.getTextSize(conf_distance_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Starting position for the second line (conf_distance_text)
        text_x = int(obj.bbox[0])
        text_y = int(obj.bbox[1]) - padding

        # Calculate the y position for the first line (label_text)
        label_y = text_y - conf_height - padding

        # Ensure both lines are within the frame boundaries
        if label_y - label_height < 0:
            # Not enough space above, place text below the bounding box
            label_y = int(obj.bbox[3]) + label_height + padding
            conf_distance_y = label_y + conf_height + padding
        else:
            conf_distance_y = text_y

        # Draw label with background
        put_text_with_background(frame, label_text, (text_x, label_y), font_scale, text_color=(0, 0, 0),
                                 bg_color=color, thickness=thickness)

        # Draw confidence and distance with background
        put_text_with_background(frame, conf_distance_text, (text_x, conf_distance_y), font_scale,
                                 text_color=(0, 0, 0), bg_color=color, thickness=thickness)


def define_lane_area(frame, center_line: LineSegment, right_line: LineSegment, alpha=0.2, mask_color=(0, 204, 119)):
    lane_roi_points = np.array([
        (center_line.upper_x, center_line.upper_y),
        (right_line.upper_x, right_line.upper_y),
        (right_line.lower_x, right_line.lower_y),
        (center_line.lower_x, center_line.lower_y)
    ], np.int32)
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
    height, width = frame.shape[:2]
    font_scale = min(width, height) / 1000
    put_text(frame, f'Behaviour: {behaviour}', position=(10, 30), font_scale=font_scale, text_color=(0, 0, 255))


def visualize_data(video_info: VideoInfo, data: PipeData, raw_frame: np.ndarray) -> np.ndarray:
    # Make a copy of the raw frame to draw on, as the original one is read-only due to shared memory
    frame = raw_frame.copy()

    car_position = (int(video_info.width / 2), video_info.height)
    draw_car_position(frame, car_position)

    cv2.line(frame, (0, 600), (frame.shape[1], 600), color=(255, 255, 255), thickness=3)

    # Calculate dynamic font_scale based on frame size if not provided
    height, width = frame.shape[:2]
    font_scale = min(width, height) / 1000  # Adjust denominator as needed for scaling
    match video_info.width:
        case n if 1920 < n:
            text_thickness = 3
        case n if 1280 <= n <= 1920:
            text_thickness = 2
        case n if n < 1280:
            text_thickness = 1
        case _:
            text_thickness = 1

    visualize_road_objects(frame, data.traffic_signs, color=(0, 255, 0), font_scale=font_scale, thickness=text_thickness)
    visualize_road_objects(frame, data.traffic_lights, color=(0, 0, 255), font_scale=font_scale, thickness=text_thickness)
    visualize_road_objects(frame, data.pedestrians, color=(255, 0, 0), font_scale=font_scale, thickness=text_thickness)

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

            if data.heading_error_degrees is not None:
                put_text_with_background(frame, f'Heading Error: {int(data.heading_error_degrees)}deg', position=(10, 50),
                                         font_scale=font_scale, thickness=text_thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0))
            if data.lateral_offset is not None:
                put_text_with_background(frame, f'Lateral Offset: {data.lateral_offset * 100:.0f}%', position=(10, 90),
                                            font_scale=font_scale, thickness=text_thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0))

    return frame