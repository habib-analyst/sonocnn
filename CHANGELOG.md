# Changelog

## 0.1.0 - 2026-10-28

- Initial release of SonoCNN as a standalone model package.
- Verified BreastMNIST weights in three formats: `model.safetensors`,
  the original PyTorch checkpoint (`assets/sonocnn.pt`), and ONNX
  (`sonocnn.onnx`, parity vs PyTorch measured at 9.5e-07 max logit diff).
- Every headline metric independently reproduced from the shipped artifacts
  (`examples/verify_metrics.py`); full per-class metrics in
  `assets/evaluation_results.json`.
- Inference API (`load_model`, `predict`, `predict_proba`, `preprocess`)
  with the exact training preprocessing contract.
- Demo (`examples/predict_demo.py`) including an optional ViewConsensus
  8-view consensus panel with a defer-to-human verdict.
