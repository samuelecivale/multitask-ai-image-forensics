# Multi-Task AI Image Forensics with EfficientNet

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![EfficientNet](https://img.shields.io/badge/Backbone-EfficientNet--B0-green)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Forensics-blue)
![Multi-Task](https://img.shields.io/badge/Learning-Multi--Task-purple)

A multi-task computer-vision system for jointly detecting **AI-generated images** and identifying **post-processing transformations** applied to them.

The model uses a shared EfficientNet-B0 backbone with two classification heads:

1. **Real vs AI-generated image detection**
2. **Transformation classification**

The project studies whether learning post-processing artifacts as an auxiliary task improves the robustness of synthetic-image detection.

---

## Main Result

| Model | Real / AI | Transformation |
|---|---:|---:|
| Real/Fake single-task baseline | **89.92%** | — |
| Transformation single-task baseline | — | **86.08%** |
| Multi-task, α = 0.5 | **91.17%** | **83.42%** |

The selected multi-task configuration therefore produces:

$$\mathbf{+1.25}\text{ percentage points}
$$

on the primary Real-vs-AI task, while transformation recognition decreases by

$$\mathbf{-2.67}\text{ percentage points}$$

relative to its dedicated single-task baseline.

This exposes a useful **multi-task trade-off** rather than a uniform improvement across both objectives.

---

## Problem

AI-generated image detectors often learn shortcuts tied to a particular generator or image-processing pipeline.

Real-world images, however, may undergo transformations such as:

- resizing;
- resampling;
- re-digitization;
- other post-processing operations.

The project investigates whether explicitly teaching the network to recognize these transformations can force the shared representation to learn features useful for more robust AI-image detection.

---

## Architecture

```text
                   Input image
                       │
                       ▼
              EfficientNet-B0
               shared backbone
                       │
                Shared features
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
      Real/Fake head       Transform head
         2 classes            3 classes
            │                     │
            ▼                     ▼
       Real vs AI       Processing category
```

The pretrained EfficientNet classifier is replaced by a shared feature representation followed by two independent linear heads.

---

## Multi-Task Objective

Training minimizes

$$\mathcal{L}=\alpha\mathcal{L}_{RF}+(1-\alpha)\mathcal{L}_{T}$$

where:

- $$\(\mathcal{L}_{RF}\)$$ is the Real/Fake cross-entropy loss;
- $$\(\mathcal{L}_{T}\)$$ is the transformation-classification loss;
- $$\(\alpha\)$$ controls the trade-off between the two tasks.

The experiments compare multiple values of

$$\alpha\in\{0.2,0.5,0.8\}$$

against independent single-task baselines.

---

## Training Pipeline

The experiments use:

- ImageNet-pretrained EfficientNet-B0;
- shared feature extraction;
- cross-entropy classification losses;
- Adam optimization;
- automatic mixed precision when available;
- validation-based model selection;
- controlled loss-weight ablations.

The best multi-task configuration is selected based on joint validation performance rather than only one output head.

---

## What the Experiment Shows

The auxiliary task does **not** improve every metric simultaneously.

Instead, the α=0.5 model improves the main forensic task while reducing transformation-classification accuracy.

This suggests that the shared backbone learns a representation that is more useful for distinguishing real from generated images, but the two tasks still compete for representational capacity.

That interference is itself an important result when designing multi-task forensic systems.

---

## Additional Experiments

The notebook also explores an artifact-aware multi-task variant and compares:

- standard multi-task learning;
- artifact-aware training;
- single-task baselines;
- robustness under image transformations.

This allows the project to examine both predictive performance and cross-task robustness.

---

## Repository Structure

```text
.
├── CV_project.ipynb
├── CV_project_presentation.pptx
└── README.md
```

---

## Running the Project

Open:

```text
CV_project.ipynb
```

in Jupyter or Google Colab.

The notebook contains the complete workflow:

```text
dataset loading
      ↓
preprocessing
      ↓
single-task baselines
      ↓
multi-task EfficientNet
      ↓
loss-weight ablation
      ↓
evaluation
      ↓
robustness analysis
```

Update the dataset path to point to the local copy of the project dataset before running the experiments.

---

## What This Project Demonstrates

- Deep Learning
- Computer Vision
- AI-Generated Image Detection
- Image Forensics
- EfficientNet
- Transfer Learning
- Multi-Task Learning
- Auxiliary Objectives
- Hyperparameter Ablation
- Robustness Evaluation
- PyTorch

---

## Key Takeaway

The project shows that auxiliary supervision can improve the primary AI-image forensic task even when the auxiliary classifier itself becomes less accurate.

Rather than assuming that multi-task learning universally improves all outputs, the experiments quantify the **performance trade-off introduced by shared representations**.
