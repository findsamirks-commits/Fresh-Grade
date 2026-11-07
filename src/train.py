"""Production Training Loop for Multi-Task ViT with Joint Multi-Task Loss.

Key capabilities:
- Layer-wise learning rate: backbone at 3e-5, task heads at 3e-4 (10x ratio)
- Linear warmup with cosine decay (get_cosine_schedule_with_warmup)
- Mixed-precision acceleration via torch.cuda.amp.GradScaler
- Early-best checkpoint saving evaluated on validation Macro-F1
- Comprehensive evaluation metrics:
  - Classification: Accuracy, Macro-F1, Weighted-F1, Per-class F1 breakdown
  - Shelf-life regression: MAE, RMSE, R²
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.model import FreshGradeMultiTaskViT, MultiTaskLoss
from src.data import (
    CLASSES,
    GRADE_TO_ID,
    FreshGradeDataset,
    create_synthetic_benchmark_dataset,
    transform_pipeline,
)


class CSVDataset(Dataset):
    """Dataset loading produce samples from an exported train/val/test CSV."""

    def __init__(self, csv_path: str, base_dir: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.transform = transform or transform_pipeline

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["relative_path"] if "relative_path" in row else row["image_filename"]
        img_path = os.path.join(self.base_dir, rel_path)
        image = Image.open(img_path).convert("RGB")
        tensor_img = self.transform(image)

        grade_id = int(row["grade_id"])
        days = float(row["days_to_spoil"])
        return tensor_img, torch.tensor(grade_id, dtype=torch.long), torch.tensor(days, dtype=torch.float32)


def get_lr_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Cosine learning rate scheduler with linear warmup."""
    try:
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    except ImportError:
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Runs full evaluation on validation split."""
    model.eval()
    total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
    all_preds_cls = []
    all_targets_cls = []
    all_preds_days = []
    all_targets_days = []

    with torch.no_grad():
        for images, targets_cls, targets_days in dataloader:
            images = images.to(device)
            targets_cls = targets_cls.to(device)
            targets_days = targets_days.to(device)

            logits, pred_days, _ = model(images)
            loss, loss_c, loss_r = criterion(logits, pred_days, targets_cls, targets_days)

            total_loss += loss.item() * images.size(0)
            total_cls_loss += loss_c.item() * images.size(0)
            total_reg_loss += loss_r.item() * images.size(0)

            preds_c = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds_cls.extend(preds_c)
            all_targets_cls.extend(targets_cls.cpu().numpy())

            all_preds_days.extend(pred_days.squeeze(-1).cpu().numpy())
            all_targets_days.extend(targets_days.cpu().numpy())

    num_samples = max(1, len(all_targets_cls))
    acc = accuracy_score(all_targets_cls, all_preds_cls)
    macro_f1 = f1_score(all_targets_cls, all_preds_cls, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets_cls, all_preds_cls, average="weighted", zero_division=0)
    per_class_f1 = f1_score(all_targets_cls, all_preds_cls, average=None, zero_division=0)

    mae = mean_absolute_error(all_targets_days, all_preds_days)
    rmse = math.sqrt(np.mean((np.array(all_targets_days) - np.array(all_preds_days)) ** 2))
    r2 = r2_score(all_targets_days, all_preds_days) if len(all_targets_days) > 1 else 0.0

    metrics = {
        "val_loss": total_loss / num_samples,
        "val_cls_loss": total_cls_loss / num_samples,
        "val_reg_loss": total_reg_loss / num_samples,
        "val_accuracy": acc,
        "val_macro_f1": macro_f1,
        "val_weighted_f1": weighted_f1,
        "val_days_mae": mae,
        "val_days_rmse": rmse,
        "val_days_r2": r2,
    }
    for i, c_name in enumerate(CLASSES):
        metrics[f"f1_{c_name}"] = per_class_f1[i] if i < len(per_class_f1) else 0.0

    return metrics


def train_model(
    epochs: int = 5,
    batch_size: int = 8,
    lr_backbone: float = 3e-5,
    lr_heads: float = 3e-4,
    data_dir: Optional[str] = None,
    save_dir: str = "runs/fresh_grade",
) -> Dict[str, Any]:
    """Production training harness for FreshGradeMultiTaskViT."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Initializing Fresh-Grade Training on [{device}]")

    # Datasets
    if data_dir and os.path.exists(os.path.join(data_dir, "train.csv")):
        train_csv = os.path.join(data_dir, "train.csv")
        val_csv = os.path.join(data_dir, "val.csv")
        train_dataset = CSVDataset(train_csv, base_dir=data_dir)
        val_dataset = CSVDataset(val_csv, base_dir=data_dir)
        print(f"Loaded CSV datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    else:
        print("Using synthetic multi-class produce benchmark dataset...")
        raw_train = create_synthetic_benchmark_dataset(num_samples_per_class=12)
        raw_val = create_synthetic_benchmark_dataset(num_samples_per_class=4)
        train_dataset = FreshGradeDataset(raw_train)
        val_dataset = FreshGradeDataset(raw_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FreshGradeMultiTaskViT(num_classes=len(CLASSES), pretrained=False).to(device)
    criterion = MultiTaskLoss(alpha=1.0, beta=0.5, label_smoothing=0.05)

    # Layer-wise parameter grouping
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vit" in name or "patch_embed" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 0.01},
        {"params": head_params, "lr": lr_heads, "weight_decay": 0.01},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    total_training_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_training_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_macro_f1 = -1.0
    best_weights_path = os.path.join(save_dir, "best.pt")

    print(f"Total training steps: {total_training_steps} (Warmup: {warmup_steps}) | AMP: {use_amp}")
    print("-" * 75)

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start_t = time.perf_counter()

        for images, targets_cls, targets_days in train_loader:
            images = images.to(device)
            targets_cls = targets_cls.to(device)
            targets_days = targets_days.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=use_amp):
                logits, pred_days, _ = model(images)
                loss, loss_c, loss_r = criterion(logits, pred_days, targets_cls, targets_days)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / len(train_dataset)
        val_metrics = evaluate(model, criterion, val_loader, device)
        elapsed = time.perf_counter() - start_t

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(val_metrics["val_loss"], 4),
            "val_accuracy": round(val_metrics["val_accuracy"], 4),
            "val_macro_f1": round(val_metrics["val_macro_f1"], 4),
            "val_days_mae": round(val_metrics["val_days_mae"], 2),
            "epoch_sec": round(elapsed, 1),
        }
        history.append(epoch_info)

        print(
            f"Epoch [{epoch+1:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_accuracy']*100:.1f}% | "
            f"Val Macro-F1: {val_metrics['val_macro_f1']:.3f} | "
            f"MAE: {val_metrics['val_days_mae']:.2f}d ({elapsed:.1f}s)"
        )

        f1_str = " | ".join([f"{c}: {val_metrics[f'f1_{c}']:.2f}" for c in CLASSES])
        print(f"   ↳ Per-Class F1: {f1_str}")

        if val_metrics["val_macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["val_macro_f1"]
            torch.save(model.state_dict(), best_weights_path)
            model.save_pretrained(save_dir)
            print(f"   ⭐ New best model saved! (Macro-F1: {best_val_macro_f1:.4f})")

    print("-" * 75)
    print(f"✅ Training completed! Best checkpoint at: {best_weights_path}")
    return {"history": history, "best_val_macro_f1": best_val_macro_f1}


def main():
    parser = argparse.ArgumentParser(description="Train Fresh-Grade Multi-Task ViT")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing train.csv and images/")
    parser.add_argument("--save-dir", type=str, default="runs/fresh_grade", help="Directory to save checkpoints")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
