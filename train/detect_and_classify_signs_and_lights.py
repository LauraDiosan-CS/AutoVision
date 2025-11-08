import os
from pathlib import Path

import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# -------- Paths --------
YOLO_WEIGHTS = "exp/det_bdd_v11_custom2/weights/best.pt"
SOURCE_DIR = "datasets/ntt_env"
OUTPUT_DIR = "runs/detect/predict_ntt_sign_light_class"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# -------- Load YOLO detector --------
detector = YOLO(YOLO_WEIGHTS)

# -------- Traffic SIGN classifier (GTSRB) --------
sign_proc = AutoImageProcessor.from_pretrained("kelvinandreas/vit-traffic-sign-GTSRB")
sign_cls = AutoModelForImageClassification.from_pretrained(
    "kelvinandreas/vit-traffic-sign-GTSRB", use_safetensors=True
).eval()

SIGN_ID2LABEL = {}
for k, v in sign_cls.config.id2label.items():
    try:
        SIGN_ID2LABEL[int(k)] = v
    except Exception:
        SIGN_ID2LABEL[k] = v

# -------- Classical CV-based traffic light color detector --------
def classify_light_cv(crop_pil):
    """Return ('red'|'yellow'|'green'|'none', confidence) using HSV masks."""
    crop = np.array(crop_pil.convert("RGB"))
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # HSV masks
    red1 = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 100, 80), (179, 255, 255))
    yellow = cv2.inRange(hsv, (15, 100, 80), (35, 255, 255))
    green = cv2.inRange(hsv, (45, 100, 80), (90, 255, 255))

    scores = {
        "red": float(red1.mean() + red2.mean()),
        "yellow": float(yellow.mean()),
        "green": float(green.mean()),
    }
    color = max(scores, key=scores.get)
    conf = scores[color] / 255.0
    # very low confidence → probably off / dark
    if conf < 0.05:
        color = "none"
    return color, conf

def get_font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

FONT = get_font(16)

# -------- Inference loop: detect → crop → classify --------
for result in detector.predict(source=SOURCE_DIR, imgsz=640, conf=0.25, stream=True):
    img_path = result.path
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes = result.boxes

    if boxes is None or boxes.xyxy is None:
        continue

    for box, c in zip(boxes.xyxy, boxes.cls):
        det_label = detector.names[int(c)]
        if det_label not in ("traffic light", "traffic sign"):
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        crop = img.crop((x1, y1, x2, y2))

        if det_label == "traffic sign":
            x = sign_proc(images=crop, return_tensors="pt")
            with torch.no_grad():
                y = sign_cls(**x).logits.softmax(-1)
            idx = int(y.argmax())
            conf = float(y.max())
            sign_name = SIGN_ID2LABEL.get(idx, SIGN_ID2LABEL.get(str(idx), str(idx)))
            caption = f"{det_label}: {sign_name} ({conf:.2f})"
        else:  # traffic light → OpenCV color detection
            color, conf = classify_light_cv(crop)
            caption = f"{det_label}: {color} ({conf:.2f})"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # place label BELOW box with small offset
        tw, th = draw.textbbox((0, 0), caption, font=FONT)[2:]
        pad = 2
        text_y = y2 + 4  # below bbox
        draw.rectangle(
            [x1, text_y, x1 + tw + 2 * pad, text_y + th + 2 * pad],
            fill=(0, 0, 0, 160)
        )
        draw.text((x1 + pad, text_y + pad), caption, fill="white", font=FONT)

    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    img.save(out_path)
    print("Saved:", out_path)

print("✅ Done. Annotated outputs in:", OUTPUT_DIR)