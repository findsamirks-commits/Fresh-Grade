"""Dataset ingestion, converters, and synthetic produce generator for Fresh-Grade.

Provides three raw-dataset converters:
1. convert_mendeley_banana: Maps 7 ripening stages to 4 grades and continuous days-to-spoil.
2. convert_kaggle_fresh_rotten: Ingests binary Kaggle Fresh/Rotten produce images into 4-grade distribution.
3. convert_milkbasket: Quick-commerce dark-store protocol (grade subfolders or labels.csv).

All converters output stratified train/val/test CSVs and a grade_to_id.json manifest.
"""

import json
import math
import os
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["Fresh", "Ripe", "Overripe", "Spoiled"]
GRADE_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_GRADE = {i: c for i, c in enumerate(CLASSES)}
CLASS_TO_IDX = GRADE_TO_ID  # Alias for backward compatibility

# Standard ViT image transformations (224x224, ImageNet normalization)
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# -----------------------------------------------------------------------------
# Synthetic Produce Generator
# -----------------------------------------------------------------------------
def generate_synthetic_produce_image(
    produce_type: str = "tomato",
    grade: str = "Ripe",
    output_path: Optional[str] = None,
) -> Image.Image:
    """Generates synthetic produce image exhibiting color & texture signatures for each grade."""
    img = Image.new("RGB", (320, 320), color=(241, 245, 249))
    draw = ImageDraw.Draw(img)

    bbox = [40, 40, 280, 280]

    if produce_type.lower() == "banana":
        if grade == "Fresh":
            fill_color = (132, 204, 22)   # Vibrant green-yellow
            speckles = 0
        elif grade == "Ripe":
            fill_color = (250, 204, 21)   # Bright golden yellow
            speckles = 5
        elif grade == "Overripe":
            fill_color = (161, 98, 7)     # Brown-yellow with high spotting
            speckles = 25
        else:  # Spoiled
            fill_color = (68, 64, 60)     # Dark blackish brown
            speckles = 50

        draw.arc([30, 30, 290, 290], start=30, end=170, fill=fill_color, width=65)
        draw.ellipse([215, 60, 255, 95], fill=(101, 163, 13))

    else:
        if grade == "Fresh":
            fill_color = (34, 197, 94)    # Firm unripe green
            stem_color = (22, 101, 52)
            spots = []
        elif grade == "Ripe":
            fill_color = (220, 38, 38)    # Plump vibrant crimson red
            stem_color = (22, 101, 52)
            spots = []
        elif grade == "Overripe":
            fill_color = (153, 27, 27)    # Soft deep maroon with wrinkling
            stem_color = (113, 63, 18)
            spots = [(120, 140, 15), (170, 190, 20)]
        else:  # Spoiled
            fill_color = (69, 26, 3)      # Dark decaying brown with fungal patches
            stem_color = (41, 37, 36)
            spots = [(100, 110, 35), (160, 170, 45), (130, 200, 30)]

        draw.ellipse(bbox, fill=fill_color, outline=(255, 255, 255), width=2)
        draw.polygon([(160, 20), (145, 55), (175, 55)], fill=stem_color)
        draw.polygon([(160, 50), (120, 35), (140, 65)], fill=stem_color)
        draw.polygon([(160, 50), (200, 35), (180, 65)], fill=stem_color)

        for sx, sy, srad in spots:
            draw.ellipse([sx - srad, sy - srad, sx + srad, sy + srad], fill=(168, 162, 158, 180))

    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, quality=95)

    return img


# -----------------------------------------------------------------------------
# Dataset Ingestion Converters
# -----------------------------------------------------------------------------
def export_stratified_splits(
    records: List[Dict[str, Any]],
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    resize_to: Optional[Tuple[int, int]] = (224, 224),
) -> Dict[str, str]:
    """Saves records into stratified train.csv, val.csv, test.csv and grade_to_id.json."""
    os.makedirs(output_dir, exist_ok=True)
    images_dest_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dest_dir, exist_ok=True)

    # Process and copy/resize images
    processed_records = []
    for i, r in enumerate(records):
        src_path = r["image_path"]
        ext = os.path.splitext(src_path)[1].lower() or ".jpg"
        target_name = f"produce_{i:06d}{ext}"
        target_path = os.path.join(images_dest_dir, target_name)

        if resize_to and os.path.exists(src_path):
            with Image.open(src_path) as im:
                im_rgb = im.convert("RGB").resize(resize_to, Image.Resampling.BILINEAR)
                im_rgb.save(target_path, quality=90)
        elif os.path.exists(src_path):
            shutil.copy2(src_path, target_path)

        processed_records.append({
            "image_filename": target_name,
            "relative_path": os.path.join("images", target_name),
            "grade": r["grade"],
            "grade_id": GRADE_TO_ID[r["grade"]],
            "days_to_spoil": round(float(r["days_to_spoil"]), 2),
            "sku_category": r.get("sku_category", "general_produce"),
        })

    df = pd.DataFrame(processed_records)

    # Stratified split on grade_id
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, stratify=df["grade_id"], random_state=42
    )
    val_prop = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_prop, stratify=temp_df["grade_id"], random_state=42
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    manifest_path = os.path.join(output_dir, "grade_to_id.json")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(GRADE_TO_ID, f, indent=2)

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "manifest": manifest_path,
        "total_samples": len(df),
    }


def convert_mendeley_banana(raw_dir: str, output_dir: str) -> Dict[str, str]:
    """Ingests the Mendeley Banana Ripening dataset (7 ripeness stages).

    Mapping logic:
    - Stages 1-2 (Green / Emerging) -> Grade: Fresh, Days: Normal(7.5, 1.0)
    - Stages 3-4 (Yellow with green tips / Solid Yellow) -> Grade: Ripe, Days: Normal(4.0, 0.8)
    - Stages 5-6 (Yellow with brown specks) -> Grade: Overripe, Days: Normal(1.8, 0.4)
    - Stage 7 (Decaying brown/black) -> Grade: Spoiled, Days: Normal(0.3, 0.15)
    """
    records = []
    if not os.path.exists(raw_dir):
        print(f"[Warning] Raw Mendeley directory not found at {raw_dir}")
        return {}

    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, f)
                parent = os.path.basename(root).lower()

                # Infer stage from folder or filename
                stage = 3  # default
                for s in range(1, 8):
                    if f"stage_{s}" in parent or f"stage{s}" in parent or f"s{s}" in f.lower():
                        stage = s
                        break

                if stage in (1, 2):
                    grade = "Fresh"
                    days = max(5.0, np.random.normal(7.5, 1.0))
                elif stage in (3, 4):
                    grade = "Ripe"
                    days = max(2.5, min(5.5, np.random.normal(4.0, 0.8)))
                elif stage in (5, 6):
                    grade = "Overripe"
                    days = max(1.0, min(2.5, np.random.normal(1.8, 0.4)))
                else:
                    grade = "Spoiled"
                    days = max(0.05, min(0.8, np.random.normal(0.3, 0.15)))

                records.append({
                    "image_path": full_path,
                    "grade": grade,
                    "days_to_spoil": days,
                    "sku_category": "banana",
                })

    return export_stratified_splits(records, output_dir)


def convert_kaggle_fresh_rotten(raw_dir: str, output_dir: str) -> Dict[str, str]:
    """Ingests Kaggle Fresh and Rotten Fruit/Vegetable dataset.

    Converts binary 'fresh*' and 'rotten*' folders into a granular 4-grade distribution:
    - Fresh items: mapped to 'Fresh' (~70%, days ~ N(7.0, 1.5)) or 'Ripe' (~30%, days ~ N(3.5, 0.8))
    - Rotten items: mapped to 'Overripe' (~35%, days ~ N(1.5, 0.4)) or 'Spoiled' (~65%, days ~ N(0.4, 0.2))
    """
    records = []
    if not os.path.exists(raw_dir):
        print(f"[Warning] Raw Kaggle directory not found at {raw_dir}")
        return {}

    for root, _, files in os.walk(raw_dir):
        folder_name = os.path.basename(root).lower()
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, f)
                is_fresh = "fresh" in folder_name or "fresh" in f.lower()

                if is_fresh:
                    if random.random() < 0.70:
                        grade = "Fresh"
                        days = max(5.0, np.random.normal(7.0, 1.2))
                    else:
                        grade = "Ripe"
                        days = max(2.5, min(5.0, np.random.normal(3.5, 0.8)))
                else:
                    if random.random() < 0.35:
                        grade = "Overripe"
                        days = max(1.0, min(2.2, np.random.normal(1.5, 0.3)))
                    else:
                        grade = "Spoiled"
                        days = max(0.05, min(0.8, np.random.normal(0.4, 0.18)))

                sku = "produce"
                for candidate in ["apple", "banana", "tomato", "orange", "pepper", "cucumber"]:
                    if candidate in folder_name or candidate in f.lower():
                        sku = candidate
                        break

                records.append({
                    "image_path": full_path,
                    "grade": grade,
                    "days_to_spoil": days,
                    "sku_category": sku,
                })

    return export_stratified_splits(records, output_dir)


def convert_milkbasket(raw_dir: str, output_dir: str) -> Dict[str, str]:
    """Ingests Milkbasket quick-commerce dark store audit datasets.

    Supports two structures:
    1. Subdirectories named 'Fresh', 'Ripe', 'Overripe', 'Spoiled' containing produce images.
    2. Or a 'labels.csv' index containing: image_path, grade, days_to_spoil.
    """
    records = []
    if not os.path.exists(raw_dir):
        print(f"[Warning] Raw Milkbasket directory not found at {raw_dir}")
        return {}

    labels_csv_path = os.path.join(raw_dir, "labels.csv")
    if os.path.exists(labels_csv_path):
        df_labels = pd.read_csv(labels_csv_path)
        for _, row in df_labels.iterrows():
            img_rel = row["image_path"]
            img_full = os.path.join(raw_dir, img_rel) if not os.path.isabs(img_rel) else img_rel
            grade = str(row["grade"]).capitalize()
            days = float(row.get("days_to_spoil", 3.0))

            if grade in GRADE_TO_ID and os.path.exists(img_full):
                records.append({
                    "image_path": img_full,
                    "grade": grade,
                    "days_to_spoil": days,
                    "sku_category": row.get("sku_category", "milkbasket_produce"),
                })
    else:
        # Scan grade subfolders
        days_defaults = {"Fresh": 7.0, "Ripe": 3.5, "Overripe": 1.5, "Spoiled": 0.3}
        for grade_folder in CLASSES:
            folder_path = os.path.join(raw_dir, grade_folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        records.append({
                            "image_path": os.path.join(folder_path, f),
                            "grade": grade_folder,
                            "days_to_spoil": days_defaults[grade_folder],
                            "sku_category": "darkstore_inbound",
                        })

    return export_stratified_splits(records, output_dir)


# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------
class FreshGradeDataset(Dataset):
    """In-memory or disk-backed dataset for multi-task produce quality training."""

    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform or transform_pipeline

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        tensor_img = self.transform(image)
        label_class = torch.tensor(GRADE_TO_ID[item["grade"]], dtype=torch.long)
        label_days = torch.tensor(item["days_to_spoil"], dtype=torch.float32)

        return tensor_img, label_class, label_days


def create_synthetic_benchmark_dataset(num_samples_per_class: int = 15) -> List[Dict]:
    """Generates a synthetic dataset covering all 4 produce stages with expected days-to-spoil."""
    dataset = []
    days_distribution = {
        "Fresh": (5.0, 9.0),
        "Ripe": (2.0, 4.5),
        "Overripe": (0.8, 1.9),
        "Spoiled": (0.0, 0.4),
    }

    for grade in CLASSES:
        min_d, max_d = days_distribution[grade]
        for _ in range(num_samples_per_class):
            days = round(random.uniform(min_d, max_d), 2)
            produce = random.choice(["tomato", "banana"])
            img = generate_synthetic_produce_image(produce, grade)
            dataset.append({
                "image": img,
                "produce_type": produce,
                "grade": grade,
                "days_to_spoil": days,
            })
    return dataset


if __name__ == "__main__":
    demo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "demo_produce.jpg")
    generate_synthetic_produce_image(produce_type="tomato", grade="Ripe", output_path=demo_path)
    print(f"Sample demo produce image generated at {demo_path}")
