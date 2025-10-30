import os
import json
import shutil
import random
from PIL import Image
from tqdm import tqdm

SRC = "datasets/archive"
IMAGES_DIR = f"{SRC}/bdd100k/bdd100k/images/100k/train"
LABELS_JSON = f"{SRC}/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"

OUT = "datasets/bdd100k_yolo_custom"

SPLIT_RATIOS = (0.75, 0.20, 0.05)  # train / val / test

CLS2ID = {
    "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
    "train": 5, "motorcycle": 6, "bicycle": 7,
    "traffic light": 8, "traffic sign": 9
}
IGNORED = {"lane", "drivable area", "area/drivable", "area/alternative"}

def ensure_dirs():
    for s in ["train", "val", "test"]:
        os.makedirs(f"{OUT}/images/{s}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)

def collect_labeled_frames(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labeled = []
    for frame in tqdm(data, desc="Scanning annotations"):
        name = frame["name"]
        img_path = os.path.join(IMAGES_DIR, name)
        if not os.path.exists(img_path):
            continue
        objs = []
        for lab in frame.get("labels", []):
            cat = lab.get("category")
            if cat not in CLS2ID or cat in IGNORED or "box2d" not in lab:
                continue
            objs.append(lab)
        if objs:
            labeled.append((img_path, objs))
    return labeled

def write_yolo_label(img_path, objs, split):
    img = Image.open(img_path)
    w, h = img.size
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = f"{OUT}/labels/{split}/{img_name}.txt"
    with open(label_path, "w") as f:
        for obj in objs:
            cat = obj["category"]
            b = obj["box2d"]
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw /= w
            bh /= h
            f.write(f"{CLS2ID[cat]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

def copy_and_label(split_name, items):
    for img_path, objs in tqdm(items, desc=f"Writing {split_name}"):
        img_name = os.path.basename(img_path)
        dst_img = f"{OUT}/images/{split_name}/{img_name}"
        shutil.copy(img_path, dst_img)
        write_yolo_label(img_path, objs, split_name)

def write_yaml():
    yaml_path = "data/bdd100k_custom.yaml"
    os.makedirs("data", exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            "path: datasets/bdd100k_yolo_custom\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
        )
        for k, v in CLS2ID.items():
            f.write(f"  {v}: {k}\n")
    print(f"✅ Wrote YAML: {yaml_path}")

def main():
    ensure_dirs()
    labeled = collect_labeled_frames(LABELS_JSON)
    total = len(labeled)
    if total == 0:
        print("No labeled images found. Check paths/JSON file.")
        return

    random.shuffle(labeled)
    n_train = max(1, int(total * SPLIT_RATIOS[0]))
    n_val = max(1, int(total * SPLIT_RATIOS[1]))

    train_set = labeled[:n_train]
    val_set = labeled[n_train:n_train + n_val]
    test_set = labeled[n_train + n_val:]

    print(f"Split sizes → train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    copy_and_label("train", train_set)
    copy_and_label("val", val_set)
    copy_and_label("test", test_set)

    write_yaml()
    print("\n✅ Conversion complete!")

if __name__ == "__main__":
    main()