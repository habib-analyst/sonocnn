"""SonoCNN — compact CNN for BreastMNIST breast-ultrasound classification.

A research/benchmark model release (not a medical device). See the model
card (README.md / MODEL_CARD.md) for verified metrics and limitations.
"""
from .infer import LABELS, TRAIN_MEAN, TRAIN_STD, load_model, predict, predict_proba, preprocess
from .model import SonoCNN, build_model

__version__ = "0.1.0"
__all__ = [
    "SonoCNN",
    "build_model",
    "load_model",
    "predict",
    "predict_proba",
    "preprocess",
    "LABELS",
    "TRAIN_MEAN",
    "TRAIN_STD",
]
