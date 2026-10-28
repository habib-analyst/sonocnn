"""SonoCNN demo: predict the bundled samples, a few held-out test images,
and — if the sibling ViewConsensus library is importable — an 8-view
consensus panel with a defer-to-human verdict on one test image.

This is a mechanics demo on a 28x28 benchmark model, not a clinical tool.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

from sonocnn import load_model, predict  # noqa: E402

model = load_model(os.path.join(PKG, "model.safetensors"))

print("== bundled demo samples (256x256 ultrasound stills, resized to 28x28) ==")
for name in ("benign", "malignant"):
    r = predict(model, os.path.join(PKG, "assets", "demo_samples", f"{name}.png"))
    print(f"{name}.png -> predicted {r['label']} "
          f"(confidence {r['confidence']:.1%}; benign {r['probabilities']['benign']:.3f}, "
          f"malignant {r['probabilities']['malignant']:.3f})")

print("\n== held-out BreastMNIST test images ==")
d = np.load(os.path.join(PKG, "assets", "breastmnist.npz"))
x_test = d["test_images"]
y_test = d["test_labels"].ravel()
labels = ("benign", "malignant")
for cls in (0, 1):
    idx = int(np.where(y_test == cls)[0][0])
    r = predict(model, x_test[idx])
    print(f"test[{idx}] true={labels[cls]} -> predicted {r['label']} "
          f"(confidence {r['confidence']:.1%})")

print("\n== ViewConsensus panel on the malignant test image (optional) ==")
try:
    sys.path.insert(0, os.path.join(PKG, "..", "..", "libraries", "consensuskit", "src"))
    from viewconsensus import consensus_from_probabilities

    import torch
    from sonocnn.infer import TRAIN_MEAN, TRAIN_STD

    img = x_test[int(np.where(y_test == 1)[0][0])].astype(np.float32) / 255.0

    def shift(a, dy, dx):
        return np.roll(np.roll(a, dy, axis=0), dx, axis=1)

    views = [img, img[:, ::-1], img[::-1, :], shift(img, 0, 2), shift(img, 0, -2),
             shift(img, 2, 0), shift(img, -2, 0), shift(img, 2, 2)[:, ::-1]]
    batch = torch.from_numpy(
        np.stack([((v - TRAIN_MEAN) / TRAIN_STD).astype(np.float32) for v in views]))[:, None]
    with torch.no_grad():
        probs = torch.softmax(model(batch), 1).numpy()
    res = consensus_from_probabilities(probs, labels=list(labels))
    verdict = getattr(res.verdict, "value", res.verdict)
    print(f"consensus: {res.label} at {res.probability:.1%} "
          f"(+/-{res.uncertainty:.1%} across 8 views) -> verdict: {verdict}, "
          f"should_defer: {res.should_defer}")
except ImportError:
    print("viewconsensus not importable here — install the sibling ViewConsensus "
          "library to see the consensus panel.")
