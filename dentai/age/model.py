"""Dental yaş regresyonu: ImageNet ön eğitimli ResNet18 + tek çıkışlı regresyon başlığı.

YOLO'nun regresyon başlığı yoktur; yaş tahmini bu yüzden ayrı, hafif bir CNN'dir.
Girdi: letterbox ile (256x512) oranı korunmuş, CLAHE uygulanmış panoramik röntgen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dentai.imaging import enhance_contrast, letterbox

INPUT_HW = (256, 512)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_age_model(pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, 1))
    return m


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> (3, H, W) float32, normalize edilmiş."""
    img, _ = letterbox(enhance_contrast(img_bgr), INPUT_HW, color=(0, 0, 0))
    rgb = img[..., ::-1].astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy()


class AgeEstimator:
    def __init__(self, weights: str | Path, device: str = "cpu"):
        import torch

        ckpt = torch.load(str(weights), map_location=device, weights_only=False)
        self.model = build_age_model(pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.mean, self.std = float(ckpt["age_mean"]), float(ckpt["age_std"])
        self.device = device

    def predict(self, img_bgr: np.ndarray) -> float:
        import torch

        x = torch.from_numpy(preprocess(img_bgr)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.model(x).item()
        return max(0.0, z * self.std + self.mean)
