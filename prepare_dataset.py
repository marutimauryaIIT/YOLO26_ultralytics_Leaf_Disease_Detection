import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

print("=" * 50)
print("RICE DISEASE DATASET PREPARATION")
print("=" * 50)

# === SETTINGS ===
DATASET_SOURCE = "RiceDiseases-DataSet"
OUTPUT_DIR = "rice_disease_dataset"
TRAIN_RATIO = 0.8  # 80% training, 20% validation

# Class mapping: folder name -> (class_id, label_name)
CLASS_MAP = {
    "Bacterial leaf blight": (0, "bacterial_leaf_blight"),
    "blast": (1, "blast"),
    "brownspot": (2, "brownspot"),
}

# === CREATE OUTPUT FOLDERS ===
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

print("Created output folders.")


# === CONVERT VOC XML TO YOLO FORMAT ===
def convert_voc_to_yolo(xml_path, class_id):
    """Convert Pascal VOC XML to YOLO format labels."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"  WARNING: Could not parse {xml_path}, skipping.")
        return None

    size = root.find("size")
    if size is None:
        return None

    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    if img_w == 0 or img_h == 0:
        return None

    yolo_lines = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Clamp values to image boundaries
        xmin = max(0, min(xmin, img_w))
        ymin = max(0, min(ymin, img_h))
        xmax = max(0, min(xmax, img_w))
        ymax = max(0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            continue

        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_lines if yolo_lines else None


# === PROCESS ALL IMAGES ===
all_samples = []  # List of (image_path, yolo_label_lines, class_name)

for folder_name, (class_id, class_label) in CLASS_MAP.items():
    folder_path = Path(DATASET_SOURCE) / folder_name
    if not folder_path.exists():
        print(f"  WARNING: Folder not found: {folder_path}")
        continue

    count = 0
    # Look in both 'orig' and 'rotated' subfolders (handle case differences)
    for subfolder in folder_path.iterdir():
        if not subfolder.is_dir():
            continue

        # Find all XML files
        xml_files = list(subfolder.glob("*.xml"))
        for xml_path in xml_files:
            # Find matching image file (could be .jpg, .JPG, .png, .PNG)
            stem = xml_path.stem
            img_path = None
            for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                candidate = xml_path.parent / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                continue

            # Convert XML to YOLO labels
            yolo_lines = convert_voc_to_yolo(str(xml_path), class_id)
            if yolo_lines is None:
                continue

            all_samples.append((str(img_path), yolo_lines, class_label))
            count += 1

    print(f"  {class_label}: {count} images with labels found")

# === SPLIT INTO TRAIN/VAL ===
random.seed(42)  # For reproducibility
random.shuffle(all_samples)

split_idx = int(len(all_samples) * TRAIN_RATIO)
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

print(f"\nTotal samples: {len(all_samples)}")
print(f"Training: {len(train_samples)}")
print(f"Validation: {len(val_samples)}")


# === COPY FILES TO OUTPUT ===
def save_samples(samples, split):
    for i, (img_path, yolo_lines, class_label) in enumerate(samples):
        # Create unique filename to avoid collisions
        ext = Path(img_path).suffix.lower()
        new_name = f"{class_label}_{i:04d}"

        # Copy image
        shutil.copy2(img_path, f"{OUTPUT_DIR}/images/{split}/{new_name}{ext}")

        # Write YOLO label file
        with open(f"{OUTPUT_DIR}/labels/{split}/{new_name}.txt", "w") as f:
            f.write("\n".join(yolo_lines))


save_samples(train_samples, "train")
save_samples(val_samples, "val")

# === CREATE dataset.yaml ===
# Use absolute path so YOLO can find it from anywhere
abs_path = os.path.abspath(OUTPUT_DIR).replace("\\", "/")

yaml_content = f"""# Rice Leaf Disease Dataset
# 3 classes: bacterial_leaf_blight, blast, brownspot

path: {abs_path}
train: images/train
val: images/val

names:
  0: bacterial_leaf_blight
  1: blast
  2: brownspot
"""

with open(f"{OUTPUT_DIR}/dataset.yaml", "w") as f:
    f.write(yaml_content)

# Also save a copy in the project root for easy access
with open("dataset.yaml", "w") as f:
    f.write(yaml_content)

print(f"\n{'=' * 50}")
print("DATASET READY!")
print(f"{'=' * 50}")
print(f"Output folder: {OUTPUT_DIR}/")
print("Config file:   dataset.yaml")
print("\nYou can now train with:")
print("  python train_rice_disease.py")
