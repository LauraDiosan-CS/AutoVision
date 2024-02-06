import json
import os
import cv2
import numpy as np
import math

def stack_images(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def get_roi_bbox_for_video(video_name):
    with open(os.path.join(os.path.dirname(__file__), os.path.join("config", "roi.json")), 'r') as file:
        video_data = json.load(file)
    
    if video_name in video_data:
        return video_data[video_name]
    else:
        return None  # Return None if video_name is not found

def update_roi_bbox_for_video(video_name, roi_bbox):
    file_path = os.path.join(os.path.dirname(__file__), os.path.join("config", "roi.json"))
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            video_data = json.load(file)
    else:
        video_data = {}
    
    video_data[video_name] = roi_bbox
    
    with open(file_path, 'w') as file:
        json.dump(video_data, file, indent=4)

def calculate_angle_with_Ox(line):
    for x1, y1, x2, y2 in line:
        delta_x = x2 - x1
        delta_y = y2 - y1
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
    return angle_rad, angle_deg
