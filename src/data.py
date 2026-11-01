"""Dataset preparation and synthetic produce generator for Fresh-Grade."""

import math
import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["Fresh", "Ripe", "Overripe", "Spoiled"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Standard ViT image transformations (224x224, ImageNet normalization)
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def generate_synthetic_produce_image(
    produce_type: str = "tomato",
    grade: str = "Ripe",
    output_path: Optional[str] = None,
) -> Image.Image:
    """Generates synthetic produce image exhibiting color & texture signatures for each grade."""
    img = Image.new("RGB", (320, 320), color=(241, 245, 249))
    draw = ImageDraw.Draw(img)

    # Base shape coordinates
    bbox = [40, 40, 280, 280]

    if produce_type.lower() == "banana":
        # Banana palette
        if grade == "Fresh":
            fill_color = (132, 204, 22)   # Vibrant green-yellow
            speckles = 0
        elif grade == "Ripe":
            fill_color = (250, 204, 21)   # Bright golden yellow
            speckles = 5
        elif grade == "Overripe":
            fill_color = (161, 98, 7)     # Brown-yellow with high spotting
            speckles = 25
        else: # Spoiled
            fill_color = (68, 64, 60)     # Dark blackish brown
            speckles = 50

        # Draw curved arc representation
        draw.arc([30, 30, 290, 290], start=30, end=170, fill=fill_color, width=65)
        draw.ellipse([215, 60, 255, 95], fill=(101, 163, 13))

    else:
        # Tomato palette
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
        else: # Spoiled
            fill_color = (69, 26, 3)      # Dark decaying brown with fungal patches
            stem_color = (41, 37, 36)
            spots = [(100, 110, 35), (160, 170, 45), (130, 200, 30)]

        # Draw circular tomato body
        draw.ellipse(bbox, fill=fill_color, outline=(255, 255, 255), width=2)

        # Draw calyx / stem leaves
        draw.polygon([(160, 20), (145, 55), (175, 55)], fill=stem_color)
        draw.polygon([(160, 50), (120, 35), (140, 65)], fill=stem_color)
        draw.polygon([(160, 50), (200, 35), (180, 65)], fill=stem_color)

        # Render fungal / mold patches for spoiled / overripe
        for sx, sy, srad in spots:
            draw.ellipse([sx - srad, sy - srad, sx + srad, sy + srad], fill=(168, 162, 158, 180))

    # Apply subtle blur to soften edges for realistic photographic texture
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, quality=95)

    return img


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
        label_class = torch.tensor(CLASS_TO_IDX[item["grade"]], dtype=torch.long)
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
