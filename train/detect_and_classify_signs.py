import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageDraw
import os

# === Paths ===
YOLO_WEIGHTS = "exp/det_bdd_v11_custom2/weights/best.pt"
SOURCE_DIR = "datasets/bdd100k_yolo_custom/images/test"
OUTPUT_DIR = "runs/detect/predict_signclass"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load YOLO detector ===
detector = YOLO(YOLO_WEIGHTS)

# === Load classifier ===
# proc = AutoImageProcessor.from_pretrained("bazyl/gtsrb-model")
# cls_model = AutoModelForImageClassification.from_pretrained("bazyl/gtsrb-model").eval()

proc = AutoImageProcessor.from_pretrained("kelvinandreas/vit-traffic-sign-GTSRB", use_fast=True)
cls_model  = AutoModelForImageClassification.from_pretrained("kelvinandreas/vit-traffic-sign-GTSRB", use_safetensors=True).eval()


# Fix id2label mapping (handle both int and str keys)
id2label = {}
for k, v in cls_model.config.id2label.items():
    try:
        id2label[int(k)] = v
    except Exception:
        id2label[k] = v

# === Detection + classification loop ===
for result in detector.predict(source=SOURCE_DIR, imgsz=640, conf=0.25, stream=True):
    img_path = result.path
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes = result.boxes

    if boxes is None or boxes.xyxy is None:
        continue

    for box, c in zip(boxes.xyxy, boxes.cls):
        label = detector.names[int(c)]
        if label not in ("traffic light", "traffic sign"):
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        crop = img.crop((x1, y1, x2, y2))

        if label == "traffic sign":
            # classify sign type (GTSRB model)
            x = proc(images=crop, return_tensors="pt")
            with torch.no_grad():
                y = cls_model(**x).logits.softmax(-1)
            idx = int(y.argmax())
            conf = float(y.max())
            cls_name = id2label.get(idx, id2label.get(str(idx), str(idx)))
            caption = f"{label}: {cls_name} ({conf:.2f})"
        else:
            # keep lights as-is (no light-state classifier yet)
            caption = label

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, y1 + 2), caption, fill="white")

    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    img.save(out_path)
    print("Saved:", out_path)

print("✅ Done. Annotated outputs in:", OUTPUT_DIR)