# Emotion Identification from Background Music in Cartoons – Tom and Jerry

> **Accepted at BIDA 2026** | To be published in Springer Smart Innovation, Systems and Technologies (Scopus Indexed)

A machine learning framework for classifying emotions solely from background music in animated cartoons — no speech, no dialogue, no character audio. Built on a novel hand-annotated dataset extracted from Tom and Jerry episodes, this work combines YAMNet transfer learning embeddings with handcrafted acoustic features in a leakage-free stacking ensemble, achieving **90.7% holdout accuracy** across five emotion categories.

---

## Why This Problem?

Most emotion recognition research focuses on speech — controlled vocal recordings with clear prosodic cues. Cartoon background music presents a fundamentally different challenge: rapid emotional transitions, exaggerated orchestral shifts, no lyrics, and unpredictable tempo changes. In Tom and Jerry, Scott Bradley's score *is* the emotional narrative. This work is one of the first to study emotion recognition exclusively from cartoon background music.

---

## Key Results

| Model | Holdout Accuracy |
|---|---|
| SVM (RBF kernel) | ~22–26% |
| Random Forest | ~83–87% |
| XGBoost | ~82–86% |
| **Stacking Ensemble (Proposed)** | **90.7%** |

**Class-wise Performance (Holdout Set)**

| Emotion | Precision | Recall | F1-score |
|---|---|---|---|
| Angry | 0.94 | 0.94 | 0.94 |
| Happy | 0.88 | 0.92 | 0.90 |
| Mixed | 0.86 | 0.89 | 0.88 |
| Sad | 1.00 | 0.86 | 0.92 |
| Surprised | 1.00 | 1.00 | 1.00 |

- Macro-averaged F1: **0.93**
- Weighted F1: **0.91**
- AUC: **0.97–1.00** across all classes

---

## Dataset

- **Source:** Background music extracted from multiple Tom and Jerry episodes
- **Segmentation:** 5-second clips, non-overlapping train/test episodes
- **Emotion labels:** Happy, Angry, Sad, Surprised, Mixed
- **Base samples:** 190 manually annotated clips
- **After augmentation:** 570 samples (Gaussian noise injection on training folds only)
- **Inter-rater reliability:** Cohen's Kappa κ = 0.777 (strong agreement)

The `Mixed` category captures clips with rapid emotional transitions — a defining feature of cartoon music that standard MER datasets do not address.

> **Note:** Raw audio files are not included in this repository due to copyright restrictions on Tom and Jerry episodes. The feature vectors and annotations are available in `data/`.

---

## System Pipeline

```
Raw Audio
    │
    ▼
Audio Preprocessing (16kHz, mono, amplitude normalization)
    │
    ▼
Feature Extraction
    ├── YAMNet Embeddings (1024-dim) ──┐
    └── Handcrafted Acoustic Features  ├──► Hybrid Vector (1076-dim)
        (MFCCs, Mel-spectrogram,       │
         ZCR, Spectral Centroid) ──────┘
    │
    ▼
Data Augmentation (training folds only — leakage controlled)
    │
    ▼
Leakage-Free Stacking Ensemble
    ├── Base: SVM (RBF) + Random Forest + XGBoost
    └── Meta-learner on OOF probability predictions
    │
    ▼
Emotion Prediction (5 classes)
```

---

## Feature Engineering

### YAMNet Transfer Learning Embeddings
YAMNet (MobileNetV1 pretrained on AudioSet) generates frame-level 1024-dimensional embeddings encoding timbre, rhythm, instrumentation, and acoustic texture. Frame-level embeddings are aggregated via global average pooling to produce a fixed-length clip-level representation.

### Handcrafted Acoustic Features (52-dim)
- **MFCCs** (13 coefficients): mean and standard deviation
- **Mel-spectrogram statistics**: mean and standard deviation across 12 Mel bands
- **Zero-Crossing Rate**
- **Spectral Centroid**

### Hybrid Representation
The 1024-dim YAMNet embeddings and 52-dim handcrafted features are concatenated into a 1076-dimensional hybrid vector. Ablation results confirm neither component alone is sufficient:

| Feature Set | Holdout Accuracy |
|---|---|
| Handcrafted only | 44.8% |
| YAMNet only | 51.7% |
| **Hybrid (proposed)** | **90.7%** |

---

## Methodology

### Leakage-Free Stacking Ensemble
To prevent data leakage under limited data conditions:

1. Dataset split: **85% train-validation / 15% holdout** (stratified by label)
2. Within training: **5-fold stratified cross-validation**
3. Per fold:
   - `RobustScaler` fitted on training fold only
   - `SMOTE` applied to training data only to handle class imbalance
   - Base models trained and out-of-fold (OOF) probabilities collected
4. OOF probabilities form the meta-learner's training input
5. Final predictions on holdout use averaged base model probabilities → meta-learner

**Base models:** SVM (RBF kernel), Random Forest, XGBoost — chosen for complementary inductive biases. SVM's weak standalone performance (22–26%) still contributes diverse decision boundaries that improve ensemble robustness.

---

## Repository Structure

```
├── data/
│   ├── features/          # Extracted feature vectors (NPY/CSV)
│   └── annotations/       # Emotion labels and metadata
├── src/
│   ├── preprocess.py      # Audio loading, resampling, normalization
│   ├── features.py        # YAMNet embedding extraction + handcrafted features
│   ├── augment.py         # Controlled Gaussian noise augmentation
│   ├── train.py           # Stacking ensemble training with CV
│   └── evaluate.py        # Holdout evaluation, confusion matrix, ROC curves
├── notebooks/
│   └── analysis.ipynb     # Exploratory analysis and results visualization
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `tensorflow`, `librosa`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `numpy`, `matplotlib`

### Extract Features

```bash
python src/preprocess.py --input_dir data/raw_audio/ --output_dir data/processed/
python src/features.py --input_dir data/processed/ --output_dir data/features/
```

### Train the Ensemble

```bash
python src/train.py --features data/features/ --labels data/annotations/labels.csv
```

### Evaluate on Holdout Set

```bash
python src/evaluate.py --model_path models/stacking_ensemble.pkl --holdout data/features/holdout/
```

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{kshirsagar2026emotion,
  title     = {Emotion Identification from Background Music in Cartoons -- Tom and Jerry},
  author    = {Kshirsagar, Shriya and Dusane, Tanisha and Siddheshwar, Shreya and Hajare, Aparna and Bedekar, Mangesh and Apte, Rashmi},
  booktitle = {Proceedings of BIDA 2026},
  series    = {Smart Innovation, Systems and Technologies},
  publisher = {Springer},
  year      = {2026}
}
```

---

## Authors

- Shriya Kshirsagar — MKSSS Cummins College of Engineering for Women, Pune
- **Tanisha Dusane** — MKSSS Cummins College of Engineering for Women, Pune
- Shreya Siddheshwar — MKSSS Cummins College of Engineering for Women, Pune
- Aparna Hajare — MKSSS Cummins College of Engineering for Women, Pune
- Mangesh Bedekar — MIT World Peace University, Pune
- Rashmi Apte — Koushiki Innovision, Pune

---

## Acknowledgements

We thank the BIDA 2026 program committee and Springer for accepting this work. YAMNet embeddings were sourced from TensorFlow Hub's AudioSet-pretrained model.

---

## Future Work

- Extend the dataset to a broader corpus of animated series (Looney Tunes, Pixar films)
- Explore real-time emotion tracking in streaming animated media
- Investigate the impact of exaggerated musical cues on audience emotional response
- Apply transformer-based audio models (Wav2Vec, Audio Spectrogram Transformer) for comparison
