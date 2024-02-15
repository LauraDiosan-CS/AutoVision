import cv2
import numpy as np
import math
from shapely.geometry import LineString, Point

roi = None
threshold_left = -0.5
threshold_right = 0.5
trapezoid_vertices = np.array([[(0, 0), (0, 0), (0, 0), (0, 0)]], dtype=np.int32)


def cannyEdge(frame, low_threshold, high_threshold):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, low_threshold, high_threshold)
    return edges


def houghLinesP(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def draw_lines(frame, lines, color=(255, 0, 0), thickness=2):        
    if lines is not None:
        for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

def draw_line(frame, line, color=(255, 0, 0), thickness=10):
    print('l=', line)
    for x1, y1, x2, y2 in line:
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def draw_endpoints(frame, endpoints, color=(255, 0, 0), radius=10):
    if endpoints:
        for point in endpoints:
            cv2.circle(frame, point, radius, color, thickness=-1)


def define_roi(frame, left_lane_endpoints, right_lane_endpoints):
    roi_points = np.array([left_lane_endpoints[0], left_lane_endpoints[1], right_lane_endpoints[0], right_lane_endpoints[1]], dtype=np.int32)
   
    # mask for ROI
    roi_mask = np.zeros_like(frame)
    cv2.fillPoly(roi_mask, [roi_points], (200, 0, 119))  # Set the color to red (BGR format)
   
    # combine frame with ROI mask
    frame_with_roi = cv2.bitwise_or(frame, roi_mask)
    return frame_with_roi


def calculate_angle_with_Ox(line):
    for x1, y1, x2, y2 in line:
        delta_x = x2 - x1
        delta_y = y2 - y1
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)

    return angle_rad, angle_deg

def find_intersection_point(line1_endpoints, line2_endpoints):
    line1 = LineString(line1_endpoints)
    line2 = LineString(line2_endpoints)

    intersection = line1.intersection(line2)

    #don't intersect or are collinear
    if isinstance(intersection, Point):
        intersection_coords = intersection.coords.xy
        intersection_point = (int(intersection_coords[0][0]), int(intersection_coords[1][0]))
        return intersection_point
    else:
        # Lines do not intersect or are collinear
        return None

def detect_lanes(frame, low_threshold, high_threshold, prev_left_lane_endpoints, prev_right_lane_endpoints, horizontals = False):
    global roi
    global trapezoid_vertices

    optional_frame = frame.copy()
    lanes_frame = frame.copy()
    horizontal_lines_frame = frame.copy()

    print("size:", frame.shape)
    height, width, _ = frame.shape

    y_center = int(height / 1.25)
    horizon_line = [[0, y_center, width, y_center]]
    horizon_endpoints =  [(0, y_center), (width, y_center)]
    draw_line(optional_frame, horizon_line, (255, 255, 255), 4)


    # Canny Edge Detector
    edges = cannyEdge(frame, low_threshold, high_threshold)

    if roi is None:
        height, width = edges.shape
        trapezoid_vertices = np.array([[-300, 640], [150, 350], [550, 350], [700, 640]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [trapezoid_vertices], 255)
        roi = mask

    cv2.polylines(frame, [trapezoid_vertices], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.fillPoly(frame, [trapezoid_vertices], (0, 0, 255, 0.3))

    masked_edges = cv2.bitwise_and(edges, roi)

    # Probabilistic Hough Lines
    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_length = 100
    max_line_gap = 550

    lines = houghLinesP(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw detected lines on Canny edges
    draw_lines(masked_edges, lines)

    #TODO: draw previous lanes if none were detected
    # if prev_left_lane_endpoints and prev_right_lane_endpoints:
    #     cv2.line(optional_frame, prev_left_lane_endpoints[0], prev_left_lane_endpoints[1], color=(0, 255, 0), thickness=5)
    #     cv2.line(optional_frame, prev_right_lane_endpoints[0], prev_right_lane_endpoints[1], color=(0, 255, 0), thickness=5)

    # TODO: Detecting 2 endpoints for each lane using the rectangle criteria

    left_lane_endpoints = []
    right_lane_endpoints = []

    max_left_lane = 0
    max_right_lane = 0

    min_angle = 90
    max_angle = 0
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 < width /2:
                    if abs(y1 - y2) > max_left_lane:
                        max_left_lane = abs(y1 - y2)
                elif x2 > width /2:
                    if abs(y1 - y2) > max_right_lane:
                        if abs(y1 - y2) > max_right_lane:
                            max_right_lane = abs(y1 - y2)

  
    if lines is not None:
        for line in lines:
            is_horizontal = False
            for x1, y1, x2, y2 in line:
                # check if the line is approximately vertical
                _, deg = calculate_angle_with_Ox(line)
        
                if deg < min_angle and deg > 0:
                    min_angle = deg
                if deg > max_angle:
                    max_angle = deg

                if abs(deg) < 5:
                    print('deg=', deg)
                    is_horizontal = True
                    draw_line(horizontal_lines_frame, line, (255, 0, 0))

                coords = line[0]
                #print('coords:', coords)

                if not ((abs(y1 - y2) < 50) or (abs(x1 - x2) < 100)):

                    if x2 < width /2 and abs(y1 - y2) == max_left_lane and not is_horizontal:
                        cv2.line(optional_frame, (coords[0], coords[1]), (coords[2], coords[3]), color=(0, 255, 0), thickness=5)
                        for i in range(2):
                            left_lane_endpoints.append((coords[2 * i], coords[2 * i + 1]))

                    elif x2 > width /2 and abs(y1 - y2) == max_right_lane and not is_horizontal:
                        cv2.line(optional_frame, (coords[0], coords[1]), (coords[2], coords[3]), color=(0, 255, 0), thickness=5)
                        for i in range(2):
                            right_lane_endpoints.append((coords[2 * i], coords[2 * i + 1]))

    #print("min angle:", min_angle)
    #print("max angle:", max_angle)
    if min_angle < 10:
        horizontals = True
    if left_lane_endpoints and right_lane_endpoints:
        optional_frame = define_roi(optional_frame, left_lane_endpoints, right_lane_endpoints)
        #TODO: parallelogram roi

    left_slope = None
    right_slope = None

    # calculating slopes
    if left_lane_endpoints and right_lane_endpoints:
        left_slope = (left_lane_endpoints[1][1] - left_lane_endpoints[1][0]) / (left_lane_endpoints[0][1] - left_lane_endpoints[0][0])
        right_slope = (right_lane_endpoints[1][1] - right_lane_endpoints[1][0]) / (right_lane_endpoints[0][1] - right_lane_endpoints[0][0])

        #print("left slope:", left_slope)
        #print("right slope:", right_slope)

        #TODO: 5.3.1 if slope < 0 ...

        if left_slope > threshold_left:
            cv2.putText(lanes_frame, 'right', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 200, 100), 5, cv2.LINE_AA)
        if right_slope < threshold_right:
            cv2.putText(lanes_frame, 'left', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 200, 100), 5, cv2.LINE_AA)
        if not(left_slope > threshold_left) and not(right_slope < threshold_right):
            cv2.putText(lanes_frame, 'go', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 200, 100), 5, cv2.LINE_AA)

    # Draw detected endpoints
    draw_endpoints(optional_frame, left_lane_endpoints, color=(0, 255, 0))
    draw_endpoints(optional_frame, right_lane_endpoints, color=(0, 255, 0))

    print('\nleft lane:', left_lane_endpoints)
    print('righ lane:', right_lane_endpoints)
    print('horizon:', horizon_endpoints)
    left_intersection = find_intersection_point(left_lane_endpoints, horizon_endpoints)
    #print("left int:", left_intersection)
    right_intersection = find_intersection_point(right_lane_endpoints, horizon_endpoints)
    #print("right int:", right_intersection)
    draw_endpoints(optional_frame, [left_intersection, right_intersection], color=(0, 255, 0))

    th1 = [(200, 864),(420, 864), (1520, 864), (1720, 864)]
    th2 =  [(100, 864),(520, 864), (1420, 864), (1820, 864)]
    th3 =  [(0, 864),(620, 864), (1320, 864), (1920, 864)]

    draw_endpoints(optional_frame, th1, color=(0, 255, 255))
    draw_endpoints(optional_frame, th2, color=(0, 165, 255))
    draw_endpoints(optional_frame, th3, color=(0, 0, 255))
    
    x_left_int = None
    x_right_int = None
    # steering adjustment
    if left_intersection:
        x_left_int = left_intersection[0]
    if right_intersection:
        x_right_int = right_intersection[0]

    message = ''
    
    if x_left_int and x_right_int:
        if (x_left_int >= th1[0][0] and x_left_int <= th1[1][0]) and (x_right_int >= th1[2][0] and x_right_int <= th1[3][0]):
            message = 'go'
        elif (x_left_int >= th2[0][0] and x_left_int <= th1[0][0]) and (x_right_int >= th2[2][0] and x_right_int <= th1[2][0]):
            message = 'slight left'
        elif (x_left_int >= th1[1][0] and x_left_int <= th2[1][0]) and (x_right_int >= th1[3][0] and x_right_int <= th2[3][0]):
            message = 'slight right'
        elif (x_left_int >= th3[0][0] and x_left_int <= th2[0][0]) and (x_right_int >= th3[2][0] and x_right_int <= th2[2][0]):
            message = 'hard left'
        elif (x_left_int >= th2[1][0] and x_left_int <= th3[1][0]) and (x_right_int >= th2[3][0] and x_right_int <= th3[3][0]):
            message = 'hard right'
    elif x_left_int:
        if (x_left_int >= th1[0][0] and x_left_int <= th1[1][0]):
            message = 'go'
        elif (x_left_int >= th2[0][0] and x_left_int <= th1[0][0]):
            message = 'slight left'
        elif (x_left_int >= th1[1][0] and x_left_int <= th2[1][0]):
            message = 'slight right'
        elif (x_left_int >= th3[0][0] and x_left_int <= th2[0][0]):
            message = 'hard left'
        elif (x_left_int >= th2[1][0] and x_left_int <= th3[1][0]):
            message = 'hard right'
    elif x_right_int:
        if (x_right_int >= th1[2][0] and x_right_int <= th1[3][0]):
            message = 'go'
        elif (x_right_int >= th2[2][0] and x_right_int <= th1[2][0]):
            message = 'slight left'
        elif (x_right_int >= th1[3][0] and x_right_int <= th2[3][0]):
            message = 'slight right'
        elif (x_right_int >= th3[2][0] and x_right_int <= th2[2][0]):
            message = 'hard left'
        elif (x_right_int >= th1[3][0] and x_right_int <= th2[3][0]):
            message = 'hard right'
    
    position = (50, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 3
    font_thickness = 10
    text_color = (0, 255, 0)
    cv2.putText(optional_frame, message, position, font, font_size, text_color, font_thickness)
 
    # Resize frames
    target_height = 300
    scale_factor = target_height / frame.shape[0]
    resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    resized_edges = cv2.resize(edges, None, fx=scale_factor, fy=scale_factor)
    resized_masked_edges = cv2.resize(masked_edges, None, fx=scale_factor, fy=scale_factor)
    resized_optional_frame = cv2.resize(optional_frame, None, fx=scale_factor, fy=scale_factor)
    resized_lanes_frame = cv2.resize(lanes_frame, None, fx=scale_factor, fy=scale_factor)
    resized_horizontal_lines_frame = cv2.resize(horizontal_lines_frame, None, fx=scale_factor, fy=scale_factor)

    # Frames for original, Canny edges, Hough lines and Detected Lanes
    original_frame = resized_frame.copy()
    canny_edges_frame = cv2.cvtColor(resized_edges, cv2.COLOR_GRAY2BGR)
    hough_lines_frame = cv2.cvtColor(resized_masked_edges, cv2.COLOR_GRAY2BGR)
    lane_detection_frame = resized_lanes_frame
    horizontal_lines_frame = resized_horizontal_lines_frame
    optional = resized_optional_frame

    # window with combined frames
    combined_frames = np.hstack([original_frame, canny_edges_frame, hough_lines_frame, optional])

    cv2.imshow('Lane Detection', combined_frames)
    # print("left:", left_lane_endpoints)
    # print("right:", right_lane_endpoints)

    return left_lane_endpoints, right_lane_endpoints, left_slope, right_slope, horizontals


def main():
    global roi
        
    prev_left_lane_endpoints = []
    prev_right_lane_endpoints = []

    cap = cv2.VideoCapture('C:\\Users\\Rotaru Mira\\Desktop\\CarVision\\laneDetect\\videos\\curved_lines_2024_1.avi')

    if not cap.isOpened():
        print("Error opening video file")
    else:
        cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)

        low_threshold = 150
        high_threshold = 250

        left_slopes=[]
        right_slopes=[]

        right_straight_slope = []
        left_straight_slope = []

        horizontal_ct = 0
        left_angles = []
        right_angles = []

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                prev_left_lane_endpoints, prev_right_lane_endpoints, left_slope, right_slope, horizontals = detect_lanes(frame, low_threshold, high_threshold, prev_left_lane_endpoints, prev_right_lane_endpoints)
                print('prev_left_lane_endpoints:', prev_left_lane_endpoints)
                # print('prev_right_lane_endpoints:', prev_right_lane_endpoints)
                
                if prev_left_lane_endpoints:
                    x1_left = prev_left_lane_endpoints[0][0]
                    y1_left = prev_left_lane_endpoints[0][1]
                    x2_left = prev_left_lane_endpoints[1][0]
                    y2_left = prev_left_lane_endpoints[1][1]
                
                if prev_right_lane_endpoints:
                    x1_right = prev_right_lane_endpoints[0][0]
                    y1_right = prev_right_lane_endpoints[0][1]
                    x2_right = prev_right_lane_endpoints[1][0]
                    y2_right = prev_right_lane_endpoints[1][1]

                _, left_angle = calculate_angle_with_Ox([[x1_left, y1_left, x2_left, y2_left]])
                left_angles.append(left_angle)
                
                _, right_angle = calculate_angle_with_Ox([[x1_right, y1_right, x2_right, y2_right]])
                right_angles.append(right_angle)

                if horizontals:
                    horizontal_ct += 1
                if left_slope:
                    left_slopes.append(left_slope)
                if right_slope:
                    right_slopes.append(right_slope)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.waitKey(1) & 0xFF == ord('z'):
                    left_straight_slope = left_slopes
                    right_straight_slope = right_slopes
                    left_slopes = []
                    right_slopes = []

        print('avg left = ', sum(left_angles)/len(left_angles))
        print('avg right = ', sum(right_angles)/len(right_angles))    
        print('horizontals:', horizontal_ct)
        cap.release()
        cv2.destroyAllWindows()

        if left_straight_slope and right_straight_slope:
            
            print("Curb:\n")
            print("L:", sum(left_slopes)/len(left_slopes))
            print("R:", sum(right_slopes)/len(right_slopes))

            print("Straight:\n")
            print("L:", sum(left_straight_slope)/len(left_straight_slope))
            print("R:", sum(right_straight_slope)/len(right_straight_slope))
        
    

if __name__ == '__main__':
    main()
