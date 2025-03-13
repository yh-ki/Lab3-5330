import os
import cv2
import random
import torch
import xml.etree.ElementTree as ET
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Paths
DATASET_PATH = "archive/dataset_20210629145407_top_600"
IMG_DIR = os.path.join(DATASET_PATH, "images")
ANNOT_DIR = os.path.join(DATASET_PATH, "annotations")

# Create directories for cleaned data
CLEANED_IMG_DIR = os.path.join(DATASET_PATH, "cleaned_images")
CLEANED_ANNOT_DIR = os.path.join(DATASET_PATH, "cleaned_annotations")
os.makedirs(CLEANED_IMG_DIR, exist_ok=True)
os.makedirs(CLEANED_ANNOT_DIR, exist_ok=True)

# Helper function to check annotation validity
def is_valid_annotation(xml_file, img_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img_file = root.find("filename").text
    img_path = os.path.join(img_dir, img_file)

    # Ensure image exists
    if not os.path.exists(img_path):
        return False

    # Check bounding boxes
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = [int(bbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]

        # Remove images with incorrect bounding boxes
        if xmin >= xmax or ymin >= ymax:
            return False

    return True

# Clean dataset
valid_annotations = []
i = 0
for file in os.listdir(ANNOT_DIR):
    if file.endswith(".xml"):
        xml_path = os.path.join(ANNOT_DIR, file)
        if is_valid_annotation(xml_path, IMG_DIR):
            valid_annotations.append(file)
            shutil.copy(os.path.join(IMG_DIR, file.replace(".xml", ".jpg")), CLEANED_IMG_DIR)
            shutil.copy(xml_path, CLEANED_ANNOT_DIR)
            i += 1
        if i == 1000:
            break

print(f"Valid images: {len(valid_annotations)}")

def unify_labels(annot_dir):
    for file in os.listdir(annot_dir):
        if file.endswith(".xml"):
            xml_path = os.path.join(annot_dir, file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                obj.find("name").text = "lego"  # Convert all labels to 'lego'

            tree.write(xml_path)

unify_labels(CLEANED_ANNOT_DIR)
print("All labels converted to 'lego'.")

# Paths for split datasets
SPLIT_DIRS = {
    "train": "lego_dataset/split/train",
    "val": "lego_dataset/split/val",
    "test": "lego_dataset/split/test"
}
for split in SPLIT_DIRS.values():
    os.makedirs(os.path.join(split, "images"), exist_ok=True)
    os.makedirs(os.path.join(split, "annotations"), exist_ok=True)

# Split dataset
random.shuffle(valid_annotations)
train_size = int(0.7 * len(valid_annotations))
val_size = int(0.15 * len(valid_annotations))
test_size = len(valid_annotations) - train_size - val_size

train_files = valid_annotations[:train_size]
val_files = valid_annotations[train_size:train_size + val_size]
test_files = valid_annotations[train_size + val_size:]

def move_files(file_list, split):
    for file in file_list:
        shutil.move(os.path.join(CLEANED_IMG_DIR, file.replace(".xml", ".jpg")),
                    os.path.join(SPLIT_DIRS[split], "images"))
        shutil.move(os.path.join(CLEANED_ANNOT_DIR, file),
                    os.path.join(SPLIT_DIRS[split], "annotations"))

move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

# Define dataset paths
xml_folder = "lego_dataset/split/test/annotations/"  # Folder containing XML annotations
yolo_folder = "lego_dataset/split/test/labels/"  # Folder where YOLO TXT files will be saved

# Ensure the output folder exists
os.makedirs(yolo_folder, exist_ok=True)

# Only one class ("lego")
class_name = "lego"
class_mapping = {class_name: 0}

# Convert XML to YOLO TXT
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        yolo_annotations = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Normalize coordinates
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            # Single class: "lego"
            yolo_annotations.append(f"0 {x_center} {y_center} {width} {height}")

        # Save YOLO TXT file
        txt_filename = os.path.join(yolo_folder, xml_file.replace(".xml", ".txt"))
        with open(txt_filename, "w") as f:
            f.write("\n".join(yolo_annotations))
