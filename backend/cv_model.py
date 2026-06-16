"""
cv_model.py  –  EfficientNet-based face skin condition analyser
Loads efficientnet_best.pt and runs inference on uploaded images.
"""

import io
import json
import os
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image

logger = logging.getLogger(__name__)

# ── colour-space helpers ─────────────────────────────────────────────────────
def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV (values in [0,1])."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v    = maxc
    diff = maxc - minc + 1e-9
    s    = np.where(maxc == 0, 0.0, diff / maxc)
    h    = np.zeros_like(r)
    m_r  = (maxc == r)
    m_g  = (maxc == g)
    m_b  = (maxc == b)
    h[m_r] = (g[m_r] - b[m_r]) / diff[m_r] % 6
    h[m_g] = (b[m_g] - r[m_g]) / diff[m_g] + 2
    h[m_b] = (r[m_b] - g[m_b]) / diff[m_b] + 4
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


# ── model wrapper ─────────────────────────────────────────────────────────────
class FaceScanModel:
    """Wraps EfficientNet-B0 for multi-label skin condition detection."""

    TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, model_path: str, label_map_path: str):
        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map: Dict = json.load(f)

        self.num_classes = len(self.label_map["classes"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model(model_path)
        logger.info("FaceScanModel loaded on %s  (%d classes)", self.device, self.num_classes)

    # ── private ──────────────────────────────────────────────────────────────
    def _build_model(self, path: str) -> nn.Module:
        net = models.efficientnet_b3(weights=None)
        in_features = net.classifier[1].in_features
        
        # Match the custom classifier block trained in your checkpoint
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.num_classes)
        )

        state = torch.load(path, map_location=self.device)
        # tolerate checkpoints saved as {"model": ...} or plain state-dict
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        net.load_state_dict(state, strict=False)
        net.eval()
        return net.to(self.device)

    @staticmethod
    def _denoise(img: Image.Image) -> Image.Image:
        """Light bilateral-style filter to reduce phone-camera noise."""
        import cv2
        arr = np.array(img)
        denoised = cv2.fastNlMeansDenoisingColored(arr, None, 5, 5, 7, 21)
        return Image.fromarray(denoised)

    @staticmethod
    def _normalise_lighting(img: Image.Image) -> Image.Image:
        """CLAHE on L-channel to compensate for harsh/dim lighting."""
        import cv2
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    # ── public ───────────────────────────────────────────────────────────────
    def preprocess(self, image_bytes: bytes) -> Image.Image:
        """Open, denoise, and normalise an uploaded image."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = self._denoise(img)
        img = self._normalise_lighting(img)
        return img

    def predict(self, image_bytes: bytes, threshold: float = 0.20) -> List[Dict]:
        """
        Run inference and return a sorted list of detected conditions.

        Returns
        -------
        list of dicts:  { condition, confidence, severity, ayurvedic_dosha }
        """
        img   = self.preprocess(image_bytes)
        tensor = self.TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        results = []
        for idx, prob in enumerate(probs):
            if prob >= threshold:
                condition = self.label_map["idx_to_class"][str(idx)]
                results.append({
                    "condition":       condition,
                    "confidence":      round(float(prob), 4),
                    "severity":        self._severity(float(prob)),
                    "ayurvedic_dosha": self._dosha_map(condition),
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def analyse_extra_cv(self, image_bytes: bytes) -> Dict:
        """
        Additional computer-vision biomarkers beyond the classifier:
        skin tone, redness index, texture roughness, brightness/glow.
        """
        img  = self.preprocess(image_bytes)
        arr  = np.array(img).astype(float) / 255.0
        hsv  = _rgb_to_hsv(arr)

        brightness  = float(hsv[..., 2].mean())
        saturation  = float(hsv[..., 1].mean())
        redness_idx = float(arr[..., 0].mean() - arr[..., 2].mean())  # R−B

        # texture roughness via Laplacian variance (normalised)
        import cv2
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        roughness = float(cv2.Laplacian(gray, cv2.CV_64F).var() / 10_000)
        roughness = min(roughness, 1.0)

        # perceived glow  (high brightness + low roughness = glowing)
        glow_score  = round(max(0, brightness - roughness * 0.4), 3)
        dullness    = round(1.0 - glow_score, 3)

        return {
            "brightness":    round(brightness, 3),
            "saturation":    round(saturation, 3),
            "redness_index": round(redness_idx, 3),
            "roughness":     round(roughness, 3),
            "glow_score":    glow_score,
            "dullness":      dullness,
            "skin_tone":     self._skin_tone_label(brightness),
        }

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _severity(prob: float) -> str:
        if prob >= 0.65:  return "High"
        if prob >= 0.40:  return "Moderate"
        return "Mild"

    @staticmethod
    def _dosha_map(condition: str) -> str:
        PITTA  = {"Redness", "inflammatory acne", "acne", "pigmentation", "dark spots"}
        KAPHA  = {"blackheads", "non inflammatory acne black heads",
                  "non inflammatory acne white heads", "pores"}
        VATA   = {"wrinkles"}
        if condition in PITTA:  return "Pitta"
        if condition in KAPHA:  return "Kapha"
        if condition in VATA:   return "Vata"
        return "Tridosha"

    @staticmethod
    def _skin_tone_label(brightness: float) -> str:
        if brightness > 0.75:  return "Fair"
        if brightness > 0.55:  return "Medium"
        if brightness > 0.38:  return "Wheatish"
        return "Deep"
