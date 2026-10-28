"""Verify every headline metric of the SonoCNN release from the artifacts
themselves: re-evaluate the shipped weights on the BreastMNIST test split
and compare against the metrics recorded at training time.

Protocol (identical to the original training script):
  images / 255 -> normalize with the train-split mean/std -> SonoCNN forward.

Usage:
  python examples/verify_metrics.py
      [--ckpt assets/sonocnn.pt] [--data assets/breastmnist.npz]
      [--saved-metrics assets/sonocnn_metrics.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

from sonocnn.model import build_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def evaluate(ckpt_path, data_path):
    d = np.load(data_path)
    x_train = d["train_images"].astype(np.float32) / 255.0
    x_test = d["test_images"].astype(np.float32) / 255.0
    y_test = d["test_labels"].ravel().astype(np.int64)
    mean, std = float(x_train.mean()), float(x_train.std())

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = build_model()
    model.load_state_dict(state)
    model.eval()

    x = torch.from_numpy(((x_test - mean) / std).astype(np.float32))[:, None]
    probs, preds = [], []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            out = torch.softmax(model(x[i:i + 64]), 1)
            probs += out[:, 1].tolist()
            preds += out.argmax(1).tolist()

    cm = confusion_matrix(y_test, preds).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return {
        "n_test": int(len(y_test)),
        "test_counts_benign_malignant": np.bincount(y_test, minlength=2).tolist(),
        "train_mean": round(mean, 5),
        "train_std": round(std, 5),
        "test_accuracy": round(accuracy_score(y_test, preds), 4),
        "test_auc": round(roc_auc_score(y_test, probs), 4),
        "confusion_matrix": cm,
        "benign_recall_specificity": round(tn / (tn + fp), 4),
        "malignant_recall_sensitivity": round(tp / (tp + fn), 4),
        "benign_precision": round(precision_score(y_test, preds, pos_label=0), 4),
        "malignant_precision": round(precision_score(y_test, preds, pos_label=1), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, preds), 4),
        "majority_baseline": round(float(max(np.bincount(y_test)) / len(y_test)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PKG, "assets", "sonocnn.pt"))
    ap.add_argument("--data", default=os.path.join(PKG, "assets", "breastmnist.npz"))
    ap.add_argument("--saved-metrics", default=os.path.join(PKG, "assets", "sonocnn_metrics.json"))
    ap.add_argument("--out", default=None, help="optional path to write the reproduced metrics JSON")
    args = ap.parse_args()

    res = evaluate(args.ckpt, args.data)
    print(json.dumps(res, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.out}")

    if os.path.exists(args.saved_metrics):
        saved = json.load(open(args.saved_metrics))
        print("\ncomparison vs training-time metrics file:")
        ok = True
        for k in ("test_accuracy", "test_auc", "confusion_matrix", "majority_baseline"):
            match = res[k] == saved[k]
            ok &= match
            print(f"  {k}: reproduced={res[k]}  saved={saved[k]}  match={match}")
        print("ALL HEADLINE METRICS REPRODUCED" if ok else "MISMATCH — see above")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
