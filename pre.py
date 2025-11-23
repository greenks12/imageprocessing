import pandas as pd
import os
import cv2
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt


# ======================
# 1) BASE PATH
# ======================
BASE_DIR = r"F:\Work\Project\car-dataset"

DATA_PATH = fr"{BASE_DIR}\data"


CSV_PATH = fr"{BASE_DIR}\data\train_solution_bounding_boxes (1).csv"
IMAGES_ROOT = fr"{DATA_PATH}\train"      
LABELS_DIR = fr"{BASE_DIR}\labels"

# ======================
# 2) LOAD CSV
# ======================
df = pd.read_csv(CSV_PATH)
print("CSV Loaded:", df.head())

# ======================
# 3) COLLECT ALL IMAGES
# ======================
IMAGE_EXT = ["jpg", "jpeg", "png"]
image_paths = []

for ext in IMAGE_EXT:
    image_paths += glob.glob(fr"{IMAGES_ROOT}\*.{ext}", recursive=False)

print("Found images:", len(image_paths))
if len(image_paths) == 0:
    raise SystemExit("❌ ERROR: No images found. Check IMAGES_ROOT path.")

image_map = {os.path.basename(p): p for p in image_paths}

# ======================
# 4) MAKE LABELS FOLDER
# ======================
os.makedirs(LABELS_DIR, exist_ok=True)

missing_or_bad = []

# ======================
# 5) CONVERT CSV → YOLO TXT LABELS
# ======================
for img_name, group in df.groupby("image"):

    img_path = image_map.get(img_name, None)
    if img_path is None:
        missing_or_bad.append((img_name, "not found"))
        print("❌ Missing image:", img_name)
        continue

    img = cv2.imread(img_path)
    if img is None:
        try:
            img = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]
        except Exception as e:
            print("❌ Unreadable:", img_name, e)
            continue

    h, w, _ = img.shape
    label_file = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt"))

    with open(label_file, "w") as f:
        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = row[["xmin", "ymin", "xmax", "ymax"]]

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            f.write(f"0 {x_center} {y_center} {width} {height}\n")

print("DONE label convert.")
print("Missing:", len(missing_or_bad))

# ======================
# 6) TRAIN/VAL SPLIT
# ======================
all_images = [Path(p) for p in image_paths]

train_imgs, val_imgs = train_test_split(
    all_images, test_size=0.2, random_state=42
)

OUTPUT = Path(BASE_DIR)

(OUTPUT / "images/train").mkdir(parents=True, exist_ok=True)
(OUTPUT / "images/val").mkdir(parents=True, exist_ok=True)
(OUTPUT / "labels/train").mkdir(parents=True, exist_ok=True)
(OUTPUT / "labels/val").mkdir(parents=True, exist_ok=True)

def move_split(img_list, split):
    for img in img_list:
        shutil.copy2(img, OUTPUT / f"images/{split}/{img.name}")
        label_src = Path(LABELS_DIR) / f"{img.stem}.txt"
        if label_src.exists():
            shutil.copy2(label_src, OUTPUT / f"labels/{split}/{img.stem}.txt")

move_split(train_imgs, "train")
move_split(val_imgs, "val")

print("Train/Val split completed.")

model = YOLO("yolo11s.pt")

results = model.train(
    data=r"F:\Work\Project\car-dataset\data\data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)



train_images_dir = Path("F:\Work\Project\car-dataset\data\train")
train_label_dir = Path("F:\Work\Project\car-dataset\labels\train")

image_files = list(train_images_dir.glob("*.jpg"))

valid_images = [img for img in image_files if (train_label_dir / f"{img.stem}.txt").exists()]

sample_files = random.sample(valid_images, min(9, len(valid_images)))

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
axes = axes.flatten()

for ax, img_path in zip(axes, sample_files):
    label_path = train_label_dir / f"{img_path.stem}.txt"
    image_with_boxes = plot_with_boxes(img_path, label_path)
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    
    ax.imshow(image_with_boxes)
    ax.axis("off")
    ax.set_title(img_path.name, fontsize=12)

plt.tight_layout()
plt.show()