"""Inference helpers for SonoCNN.

Preprocessing contract (identical to training/evaluation):
  grayscale -> resize to 28x28 (area interpolation) -> /255
  -> normalize with the BreastMNIST train-split mean/std
     (TRAIN_MEAN / TRAIN_STD below, computed on the train split and
     re-verified by examples/verify_metrics.py).

Labels: 0 = "benign" (normal/benign), 1 = "malignant".
"""
from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch
from PIL import Image

from .model import SonoCNN, build_model

TRAIN_MEAN = 0.32758
TRAIN_STD = 0.20573
LABELS = ("benign", "malignant")
IMG_SIZE = 28

ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


def load_model(weights_path: str | os.PathLike, device: str = "cpu") -> SonoCNN:
    """Load a SonoCNN checkpoint (release .pt or .safetensors) in eval mode."""
    model = build_model()
    path = str(weights_path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess(image: ImageLike) -> torch.Tensor:
    """One image -> (1, 1, 28, 28) normalized float tensor."""
    if isinstance(image, (str, os.PathLike)):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    img = img.convert("L")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BOX)  # BOX ~= cv2.INTER_AREA on downscale
    x = np.asarray(img).astype(np.float32) / 255.0
    x = ((x - TRAIN_MEAN) / TRAIN_STD).astype(np.float32)
    return torch.from_numpy(x)[None, None]


def predict_proba(model: SonoCNN, image: ImageLike) -> dict:
    """Return {'benign': p0, 'malignant': p1} softmax probabilities."""
    x = preprocess(image)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    return {LABELS[0]: probs[0], LABELS[1]: probs[1]}


def predict(model: SonoCNN, image: ImageLike) -> dict:
    """Return {'label', 'confidence', 'probabilities'} for one image."""
    probs = predict_proba(model, image)
    label = max(probs, key=probs.get)
    return {"label": label, "confidence": probs[label], "probabilities": probs}
