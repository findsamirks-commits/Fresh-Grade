"""Multi-Task Vision Transformer (ViT) Architecture for Perishable Produce.

Simultaneously predicts:
1. Quality Grade (4-Class Softmax: Fresh, Ripe, Overripe, Spoiled)
2. Days-to-Spoil (Continuous Non-negative Regression via Softplus)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple


class FreshGradeMultiTaskViT(nn.Module):
    """Multi-task model wrapping a Vision Transformer backbone with dual output heads."""

    def __init__(
        self,
        num_classes: int = 4,
        hidden_dim: int = 768,
        backbone_name: str = "google/vit-base-patch16-224-in21k",
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.backbone_name = backbone_name

        self.vit = None
        if pretrained:
            try:
                from transformers import ViTModel
                self.vit = ViTModel.from_pretrained(backbone_name)
            except Exception as e:
                print(f"[Warning] Could not load HuggingFace ViT weights ({e}). Initializing feature extractor.")

        # Fallback convolutional patch-embedding backbone if HF offline or running lightweight tests
        if self.vit is None:
            self.patch_embed = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        # Shared representation projection layer
        self.shared_dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        # Head 1: Produce Quality Grade Classification Head
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        # Head 2: Remaining Shelf-Life Days Regression Head (Positive scalar)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Softplus(),  # Enforces smooth positive output: days >= 0.0
        )

    def extract_features(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts [CLS] token embedding and attention map if available."""
        if self.vit is not None:
            outputs = self.vit(pixel_values=pixel_values, output_attentions=True)
            cls_token = outputs.last_hidden_state[:, 0, :]
            attention = outputs.attentions[-1] if outputs.attentions else None
            return cls_token, attention
        else:
            feat = self.patch_embed(pixel_values)
            cls_token = torch.flatten(feat, 1)
            return cls_token, None

    def forward(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values: Input image tensor [batch_size, 3, 224, 224]

        Returns:
            logits: Classification logits [batch_size, num_classes]
            days: Estimated shelf-life days [batch_size, 1]
            shared_repr: Projected embedding [batch_size, hidden_dim]
        """
        cls_token, _ = self.extract_features(pixel_values)
        shared_repr = self.shared_dense(cls_token)

        logits = self.classifier_head(shared_repr)
        days = self.regression_head(shared_repr)

        return logits, days, shared_repr


class JointLoss(nn.Module):
    """Joint Multi-Task Loss: α * CE(grade) + β * MSE(log(1 + days))."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_days: torch.Tensor,
        target_class: torch.Tensor,
        target_days: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Cross-entropy classification loss
        loss_cls = self.ce_loss(pred_logits, target_class)

        # Log1p MSE loss for shelf-life regression to penalize relative percentage errors fairly
        log_pred_days = torch.log1p(pred_days.squeeze(-1))
        log_target_days = torch.log1p(target_days)
        loss_reg = self.mse_loss(log_pred_days, log_target_days)

        total_loss = (self.alpha * loss_cls) + (self.beta * loss_reg)
        return total_loss, loss_cls, loss_reg
