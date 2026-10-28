---
license: mit
pipeline_tag: image-classification
language: en
tags:
  - medical-imaging
  - breast-ultrasound
  - medmnist
  - pytorch
  - onnx
  - safetensors
---

# SonoCNN — BreastMNIST breast-ultrasound classifier

**A compact CNN (287,330 parameters) for benign-vs-malignant classification of 28×28 breast-ultrasound images, released with something most model uploads don't have: a script that re-derives every headline metric from the shipped weights and data, so the numbers below are checkable, not just claimed.**

- Test accuracy **0.8718** and ROC AUC **0.8709** on the held-out BreastMNIST test split (156 images), vs a 0.7308 majority-class baseline — independently reproduced from the artifacts in this repo on 2026-10-28 (see *Verify it yourself*).
- Weights in three formats: `model.safetensors`, the original PyTorch checkpoint (`assets/sonocnn.pt`), and ONNX (`sonocnn.onnx`, parity vs PyTorch measured at 9.5e-07 max logit difference).
- Trained from scratch by Habib Ur Rehman for the SonoSight explainable-ultrasound project (OpenCV AI Competition 2026 / Bolt Builders Week).

> **Research and benchmark use only. This is a 28×28 benchmark model trained on 546 images. It is not a medical device, it has never been evaluated on clinical data at native resolution, and it must not be used for diagnosis or screening.**

## Verified results — BreastMNIST test split (n = 156)

| Metric | Value | Source |
|---|---|---|
| Accuracy | **0.8718** (136/156) | reproduced 2026-10-28 |
| ROC AUC | **0.8709** | reproduced 2026-10-28 |
| Balanced accuracy | 0.8145 | reproduced 2026-10-28 |
| Malignant recall (sensitivity) | 0.9386 (107/114) | reproduced 2026-10-28 |
| Benign recall (specificity) | **0.6905** (29/42) | reproduced 2026-10-28 |
| Malignant precision | 0.8917 (107/120) | reproduced 2026-10-28 |
| Benign precision | 0.8056 (29/36) | reproduced 2026-10-28 |
| Confusion matrix | [[29, 13], [7, 107]] (rows: true benign, true malignant) | reproduced 2026-10-28 |
| Majority-class baseline | 0.7308 | reproduced 2026-10-28 |
| Best validation accuracy | 0.8718 (68/78) | recorded at training time |

Read the weak row honestly: **the model is much better at catching malignant cases (93.9%) than at clearing benign ones (69.1%)** — 13 of 42 benign test images are flagged malignant. On a test set this small, the 95% confidence interval on accuracy is roughly ±5 points (≈0.81–0.92). The full reproduced record is in `assets/evaluation_results.json`; the training-time record is in `assets/sonocnn_metrics.json`; they agree exactly on every headline number.

## Quickstart

```bash
pip install "git+https://github.com/habib-analyst/sonocnn.git"
```

```python
from sonocnn import load_model, predict

model = load_model("model.safetensors")   # or assets/sonocnn.pt
result = predict(model, "assets/demo_samples/benign.png")
# {'label': 'benign', 'confidence': 0.993,
#  'probabilities': {'benign': 0.993, 'malignant': 0.007}}
```

Preprocessing contract (identical to training): grayscale → resize to 28×28 → /255 → normalize with the BreastMNIST train-split mean/std (**0.32758 / 0.20573**). Labels: `0 = benign` (normal/benign), `1 = malignant`.

ONNX: `sonocnn.onnx` takes the same normalized `(N, 1, 28, 28)` float tensor and returns logits.

## Verify it yourself

```bash
python examples/verify_metrics.py
```

This re-evaluates the shipped checkpoint on the bundled test split (`assets/breastmnist.npz`) and compares against the training-time metrics file. On 2026-10-28 it printed `ALL HEADLINE METRICS REPRODUCED`. Reproduction protocol matters: an earlier audit of this checkpoint briefly "failed" at 0.73 accuracy because the evaluator forgot the /255 step — the script exists so nobody has to take the card's word for anything.

```bash
python examples/predict_demo.py   # bundled samples + optional ViewConsensus panel
python -m pytest -q               # 8 tests: formats agree, metrics files agree, parity
```

The demo also composes SonoCNN with the sibling **ViewConsensus** library: on the first malignant test image, the single-view prediction is *malignant at 70.2%*, but the 8-view consensus panel reports *76.9% ±18.9%* and returns an **unsure — defer-to-human** verdict. A confident single number is not the whole story; the spread is information.

## Model details

- **Architecture:** 3 conv blocks (32→64→128 channels, BatchNorm, ReLU, max-pool, dropout) + 2-layer MLP head. 287,330 trainable parameters; 1.2 MB in safetensors.
- **Training:** from scratch on the BreastMNIST train split (546 images), 40 epochs, AdamW (lr 3e-3, weight decay 1e-4, cosine schedule), batch 32, class-weighted cross-entropy for the 73/27 malignant/benign imbalance; checkpoint selected by best validation accuracy. Recorded training time: 65.7 s on CPU.
- **Augmentation:** horizontal flips, ±3 px translations (edge-replicated), small Gaussian noise. The translation augmentation was added after Grad-CAM showed an earlier model fixating on the dark bottom border of BreastMNIST images instead of lesion regions — accuracy was traded (~0.6 pt) for attention on the lesion.
- **Explainability:** the original build includes Grad-CAM overlays; confidences are softmax outputs and are known to be overconfident on easy cases.

## What SonoCNN is not

- **Not a clinical tool.** Native-resolution ultrasound (e.g. the bundled 256×256 stills) is out of distribution for this model: the bundled malignant still lands at p(malignant) = 0.499 — a coin flip — after resizing to 28×28, even though its native 28×28 test image scores 0.702. The demo prints this rather than hiding it.
- **Not state of the art.** It is a small from-scratch CNN on a small benchmark, released because its numbers are verified and its failure modes are documented, not because it tops a leaderboard.
- **Not trained on private or clinical data.** Only the public BreastMNIST benchmark (below).

## Data and attribution

BreastMNIST is part of **MedMNIST v2** (Yang et al., *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data 2023), derived from the BUSI breast-ultrasound dataset (Al-Dhabyani et al., 2020). The dataset is licensed **CC BY 4.0**; `assets/breastmnist.npz` is redistributed here for metric reproduction, with attribution. The MIT license of this repository covers the code and the trained weights, not the dataset.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | Release weights (safetensors) |
| `sonocnn.onnx` / `sonocnn.onnx.data` | ONNX export (logits) |
| `config.json` | Architecture + preprocessing config |
| `src/sonocnn/` | `model.py` (architecture), `infer.py` (loading + prediction API) |
| `examples/predict_demo.py` | Inference demo incl. optional ViewConsensus panel |
| `examples/verify_metrics.py` | Full metric reproduction from the artifacts |
| `assets/sonocnn.pt` | Original training checkpoint (state dict + metrics) |
| `assets/sonocnn_metrics.json` | Training-time metrics record |
| `assets/evaluation_results.json` | Independently reproduced metrics (2026-10-28) |
| `assets/breastmnist.npz` | BreastMNIST splits (CC BY 4.0, see attribution) |
| `assets/demo_samples/` | Two 256×256 ultrasound stills used by the demo |

## License

MIT for code and weights (see `LICENSE`). Dataset under CC BY 4.0 as above.
