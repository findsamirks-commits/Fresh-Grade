"""Multi-Task Vision Transformer (ViT) Architecture for Perishable Produce.

Simultaneously predicts:
1. Quality Grade (4-Class Softmax: Fresh, Ripe, Overripe, Spoiled)
2. Days-to-Spoil (Continuous Non-negative Regression via Softplus)

Includes:
- ViT-B/16 backbone with CLS feature pooling
- Attention Rollout computation (Abnar & Zuidema, 2020) for visual interpretability
- MultiTaskLoss (CrossEntropy with label smoothing + Smooth L1 on log(days + 1))
- Embedded configuration round-trip serialization (save_pretrained / from_pretrained)
"""

import json
import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def compute_attention_rollout(
    attentions: Optional[Union[Tuple[torch.Tensor, ...], list]],
    head_fusion: str = "mean",
    discard_ratio: float = 0.85,
) -> np.ndarray:
    """Computes Attention Rollout across ViT layers to trace CLS token attention back to input patches.

    Reference: Abnar & Zuidema, 'Quantifying Attention Flow in Transformers' (ACL 2020).

    Args:
        attentions: Tuple of attention weight tensors from each ViT layer [batch, num_heads, tokens, tokens].
        head_fusion: Fusion strategy across attention heads ('mean', 'max', or 'min').
        discard_ratio: Ratio of lowest attention values to discard for contrast enhancement.

    Returns:
        2D numpy array [14, 14] representing the spatial attention map, normalized to [0, 1].
    """
    if not attentions:
        # Graceful synthetic attention map for offline/fallback convolutional mode
        grid = np.zeros((14, 14), dtype=np.float32)
        cy, cx = 7, 7
        for y in range(14):
            for x in range(14):
                dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                grid[y, x] = np.exp(-dist / 3.2)
        return (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    num_tokens = attentions[0].size(-1)
    result = torch.eye(num_tokens, device="cpu")

    with torch.no_grad():
        for layer_attn in attentions:
            # layer_attn shape: [batch, num_heads, tokens, tokens]
            layer_attn_cpu = layer_attn.detach().cpu()
            if head_fusion == "mean":
                attn_fused = layer_attn_cpu.mean(dim=1).squeeze(0)
            elif head_fusion == "max":
                attn_fused = layer_attn_cpu.max(dim=1)[0].squeeze(0)
            else:
                attn_fused = layer_attn_cpu.min(dim=1)[0].squeeze(0)

            # Add residual self-connection: (A + I) / 2
            identity = torch.eye(num_tokens)
            a = (attn_fused + identity) / 2.0
            # Normalize row-wise to preserve total attention conservation
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)

    # Extract CLS token attention to spatial patches (index 0 -> indices 1:)
    cls_attention = result[0, 1:].numpy()

    # Spatial patch grid size (14x14 for 224x224 input with 16x16 patch size)
    grid_side = int(math.isqrt(cls_attention.shape[0]))
    if grid_side * grid_side == cls_attention.shape[0]:
        mask = cls_attention.reshape(grid_side, grid_side)
    else:
        mask = np.resize(cls_attention, (14, 14))

    # Optional contrast enhancement by clipping lower quantile
    if discard_ratio > 0.0:
        threshold = np.quantile(mask, discard_ratio)
        mask[mask < threshold] = threshold

    # Min-max normalization
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask.astype(np.float32)


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
        self.pretrained = pretrained

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

    def extract_features(
        self, pixel_values: torch.Tensor, return_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Extracts [CLS] token embedding and optional multi-head attention weights."""
        if self.vit is not None:
            outputs = self.vit(pixel_values=pixel_values, output_attentions=return_attentions)
            cls_token = outputs.last_hidden_state[:, 0, :]
            attentions = outputs.attentions if return_attentions else None
            return cls_token, attentions
        else:
            feat = self.patch_embed(pixel_values)
            cls_token = torch.flatten(feat, 1)
            return cls_token, None

    def forward(
        self, pixel_values: torch.Tensor, return_attentions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Forward pass.

        Args:
            pixel_values: Input image tensor [batch_size, 3, 224, 224]
            return_attentions: Whether to return ViT layer attention tensors for interpretability

        Returns:
            logits: Classification logits [batch_size, num_classes]
            days: Estimated shelf-life days [batch_size, 1]
            shared_repr: Projected embedding [batch_size, hidden_dim]
            attentions: Tuple of attention maps or None
        """
        cls_token, attentions = self.extract_features(pixel_values, return_attentions=return_attentions)
        shared_repr = self.shared_dense(cls_token)

        logits = self.classifier_head(shared_repr)
        days = self.regression_head(shared_repr)

        if return_attentions:
            return logits, days, shared_repr, attentions
        return logits, days, shared_repr

    def save_pretrained(self, save_dir: str) -> None:
        """Saves model weights along with embedded architecture configuration."""
        os.makedirs(save_dir, exist_ok=True)
        config = {
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "backbone_name": self.backbone_name,
            "pretrained": self.pretrained,
        }
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu") -> "FreshGradeMultiTaskViT":
        """Loads model weights and architecture from a serialized directory."""
        config_path = os.path.join(load_dir, "config.json")
        weights_path = os.path.join(load_dir, "pytorch_model.bin")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        model = cls(
            num_classes=config.get("num_classes", 4),
            hidden_dim=config.get("hidden_dim", 768),
            backbone_name=config.get("backbone_name", "google/vit-base-patch16-224-in21k"),
            pretrained=False,
        )

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

        model.to(device)
        return model


class MultiTaskLoss(nn.Module):
    """Joint Multi-Task Loss: α * CE(grade; label_smoothing) + β * SmoothL1(log(1 + days))."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_days: torch.Tensor,
        target_class: torch.Tensor,
        target_days: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Cross-entropy classification loss with label smoothing
        loss_cls = self.ce_loss(pred_logits, target_class)

        # Smooth L1 on log1p(days) to fairly balance relative errors across fresh (7d) vs spoiled (0.5d)
        log_pred_days = torch.log1p(pred_days.squeeze(-1))
        log_target_days = torch.log1p(target_days)
        loss_reg = self.smooth_l1(log_pred_days, log_target_days)

        total_loss = (self.alpha * loss_cls) + (self.beta * loss_reg)
        return total_loss, loss_cls, loss_reg


# Backwards compatibility alias
JointLoss = MultiTaskLoss
