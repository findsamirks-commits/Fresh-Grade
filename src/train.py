"""Training loop for Multi-Task ViT with Joint Loss."""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import FreshGradeMultiTaskViT, JointLoss
from data import FreshGradeDataset, create_synthetic_benchmark_dataset, CLASSES


def train_model(epochs: int = 5, batch_size: int = 8, lr: float = 1e-4, save_dir: str = "runs/fresh_grade"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Fresh-Grade Multi-Task ViT on device: {device}")

    # Build synthetic training set
    raw_data = create_synthetic_benchmark_dataset(num_samples_per_class=12)
    dataset = FreshGradeDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FreshGradeMultiTaskViT(num_classes=len(CLASSES), pretrained=False).to(device)
    criterion = JointLoss(alpha=1.0, beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, labels_cls, labels_days) in enumerate(dataloader):
            images = images.to(device)
            labels_cls = labels_cls.to(device)
            labels_days = labels_days.to(device)

            optimizer.zero_grad()
            logits, pred_days, _ = model(images)
            loss, loss_c, loss_r = criterion(logits, pred_days, labels_cls, labels_days)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} (Cls: {loss_c.item():.4f}, Reg: {loss_r.item():.4f})")

    save_path = os.path.join(save_dir, "best.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Trained weights saved to {save_path}")


if __name__ == "__main__":
    train_model(epochs=3)
