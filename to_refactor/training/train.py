from ultralytics import YOLO
import torch
from generate_paths import *

def train():
    "Training object detection for trafficSigns dataset using yolov8-nano"

    generate_all_paths()
    model = YOLO('models\\yolov8n.pt')  # load a pretrained model (recommended for training)
    if torch.cuda.is_available():
        model.cuda()
        print('running on gpu started...')
    results = model.train(data='configs\\traffic_signs.yaml', epochs=100, batch=100, imgsz=640, save_dir='results\\traffic-signs-detection')


if __name__ == '__main__':
    train()
