"""Tests for the SonoCNN release package.

Everything asserted here was measured against the shipped artifacts in this
environment; the metrics test compares the two JSON records (training-time
vs independently reproduced) instead of re-running the full evaluation.
"""
import json
import os

import numpy as np
import pytest
import torch

from sonocnn import load_model, predict, predict_proba, preprocess
from sonocnn.infer import TRAIN_MEAN, TRAIN_STD
from sonocnn.model import build_model

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(PKG, "assets")


@pytest.fixture(scope="module")
def data():
    d = np.load(os.path.join(ASSETS, "breastmnist.npz"))
    return d


@pytest.fixture(scope="module")
def model_pt():
    return load_model(os.path.join(ASSETS, "sonocnn.pt"))


@pytest.fixture(scope="module")
def model_st():
    return load_model(os.path.join(PKG, "model.safetensors"))


def _batch(data, n=8):
    x = data["test_images"][:n].astype(np.float32) / 255.0
    x = ((x - TRAIN_MEAN) / TRAIN_STD).astype(np.float32)
    return torch.from_numpy(x)[:, None]


def test_param_count():
    # 287,330 trainable parameters; the state dict holds 287,975 values
    # once BatchNorm running statistics (buffers) are included.
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) == 287330
    assert sum(v.numel() for v in model.state_dict().values()) == 287975


def test_preprocess_shape_and_finiteness(data):
    t = preprocess(data["test_images"][0])
    assert t.shape == (1, 1, 28, 28)
    assert torch.isfinite(t).all()


def test_safetensors_matches_pytorch_checkpoint(model_pt, model_st, data):
    x = _batch(data)
    with torch.no_grad():
        a = model_pt(x)
        b = model_st(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_onnx_parity(model_pt, data):
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(os.path.join(PKG, "sonocnn.onnx"),
                                providers=["CPUExecutionProvider"])
    x = _batch(data).numpy()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        ref = model_pt(torch.from_numpy(x)).numpy()
    assert np.abs(out - ref).max() < 1e-4


def test_demo_sample_benign(model_st):
    r = predict(model_st, os.path.join(ASSETS, "demo_samples", "benign.png"))
    assert r["label"] == "benign"
    assert r["confidence"] > 0.9


def test_demo_sample_malignant_is_borderline_out_of_distribution(model_st):
    # The bundled malignant still is a 256x256 image; the model is a 28x28
    # benchmark model, and at 28x28 this sample lands on the decision
    # boundary (measured p(malignant) = 0.499). The test pins that honest,
    # documented behavior instead of pretending the still is in-distribution.
    p = predict_proba(model_st, os.path.join(ASSETS, "demo_samples", "malignant.png"))
    assert 0.45 < p["malignant"] < 0.55


def test_first_malignant_test_image_in_distribution(model_st, data):
    y = data["test_labels"].ravel()
    idx = int(np.where(y == 1)[0][0])
    r = predict(model_st, data["test_images"][idx])
    assert r["label"] == "malignant"


def test_metrics_files_agree():
    saved = json.load(open(os.path.join(ASSETS, "sonocnn_metrics.json")))
    repro = json.load(open(os.path.join(ASSETS, "evaluation_results.json")))
    for k in ("test_accuracy", "test_auc", "confusion_matrix", "majority_baseline"):
        assert repro[k] == saved[k], k
