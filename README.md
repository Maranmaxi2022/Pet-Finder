# PetFinder Pawpularity - Image Only Regression (SVR + ViT)
Predict a pet photo’s **Pawpularity** (0–100) from the image alone. The goal is to build a strong, reproducible pipeline that helps surface engaging shelter photos and, in turn, improves adoption outcomes.

## Highlights
* **Image‑only** approach (no tabular metadata).
* Two complementary parts:
  1. **Pretrained image features → SVR stacks** (fast, strong, low‑variance).
  2. **Fine‑tuned vision transformers** for direct regression.
* **Blending** on out‑of‑fold (OOF) predictions with non‑negative, sum‑to‑1 weights.
* Reproducible **stratified K‑Fold CV** and simple post‑processing (prediction clipping; optional scale tweak).

## Data
* **Dataset:** [Kaggle](https://www.kaggle.com/competitions/petfinder-pawpularity-score/overview) containing training images and a separate test set.
* **Target:** `Pawpularity` (integer 0–100). We treat this as a regression problem.


## Reproducible cross‑validation
* **Folds:** stratified K‑Fold on binned targets (e.g., 20 folds). The same folds are reused across SVR and fine‑tuned models.
* **OOF:** every model logs OOF predictions for consistent validation and downstream blending.
Post‑processing (simple but useful):
* **Training‑time target clipping:** cap targets to `[1, 85]` for SVR fits.
* **Prediction clipping:** cap predictions to `[1, 100]` at inference.
* **Optional scale tweak:** a tiny multiplicative scaling (e.g., ×1.032) can reduce RMSE in some setups.

## Results
* **Metric:** Root Mean Squared Error (RMSE). Lower is better.
* **Validation:** OOF on stratified folds to mirror the test distribution.

| Component                                | Example OOF CV RMSE |
| ---------------------------------------- | ------------------: |
| CLIP ViT‑B/32 + Ridge (baseline)         |             \~20–21 |
| ImageNet/timm embeddings + SVR (several) |             \~17–18 |
| ViT fine‑tune (single backbone)          |             \~17–18 |
| **Blended (SVR stacks + fine‑tunes)**    |         **\~17.06** |

## Troubleshooting
* **cuML not available?** The code paths fall back to scikit‑learn SVR (slower; may require smaller feature banks).
* **OOM during embedding extraction?** Reduce image resolution or the number of concurrent backbones.
* **CV unstable?** Increase folds or use coarser target bins when stratifying.
* **LB gap vs CV?** Try the clipping/scale tweaks and ensure OOF IDs/folds exactly match the blend inputs.