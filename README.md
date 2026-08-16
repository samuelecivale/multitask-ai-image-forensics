# Multi-Task AI Image Forensics

Deep-learning system for the **joint detection of AI-generated images and post-processing transformations**, designed to study the robustness of synthetic-image detectors under realistic image alterations.

The project uses an **EfficientNet-B0** backbone with a multi-task architecture that simultaneously predicts:

1. whether an image is **real or AI-generated**;
2. which **post-processing transformation** has been applied.

The selected model achieved **91.17% accuracy on real-vs-AI image classification**.

---

## Motivation

AI-generated image detection is increasingly relevant as generative models produce images that are progressively harder to distinguish from real photographs.

A practical limitation of many detectors is that images encountered in real-world scenarios are rarely identical to the original generated samples.

They may have been:

* compressed;
* resized;
* blurred;
* sharpened;
* filtered;
* or otherwise post-processed.

These transformations can alter the low-level visual artifacts used by synthetic-image detectors.

This project investigates whether explicitly learning to recognize post-processing transformations alongside the real/fake classification task can help the network learn richer and more robust visual representations.

---

## Problem Formulation

Given an input image (x), the model jointly solves two classification tasks.

### Task 1 — Image Origin Classification

Predict whether the image is:

```text
Real
or
AI-Generated
```

### Task 2 — Transformation Classification

Predict the post-processing operation associated with the image.

The two tasks share the same visual backbone but use separate classification heads.

---

## Architecture

The model is based on **EfficientNet-B0**, used as a shared feature extractor.

```text
                         ┌──────────────────────┐
                         │     Input Image      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   EfficientNet-B0    │
                         │  Shared Backbone     │
                         └──────────┬───────────┘
                                    │
                         Shared Feature Vector
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌───────────────────┐               ┌─────────────────────┐
        │ Real / AI Head    │               │ Transformation Head │
        └─────────┬─────────┘               └──────────┬──────────┘
                  │                                    │
                  ▼                                    ▼
            Real / Fake                         Transformation
            Prediction                           Prediction
```

This setup encourages the backbone to learn features useful for both image provenance and transformation recognition.

---

## Multi-Task Learning

The network is optimized using a joint objective:

$$ \mathcal{L}_{\text{total}} = \lambda_{rf}\mathcal{L}_{\text{real/fake}} +\lambda_{tr}\mathcal{L}_{\text{transformation}} $$

where:

- $\mathcal{L}_{\text{real/fake}}$ is the classification loss for image origin;
- $\mathcal{L}_{\text{transformation}}$ is the transformation-classification loss;
- $\lambda_{rf}$ and $\lambda_{tr}$ control the contribution of the two objectives.

The key idea is that transformation recognition acts as an auxiliary task, potentially forcing the backbone to capture information about how image statistics change after post-processing.

---

## Dataset

Experiments were conducted using the **RRDataset**, containing real and AI-generated images together with post-processing variants.

The evaluation focuses on two related questions:

1. **Can the model reliably distinguish real images from AI-generated ones?**
2. **Can it simultaneously identify transformations applied to those images?**

The dataset is divided into training, validation, and test subsets to evaluate generalization on unseen samples.

---

## Experimental Pipeline

The overall workflow is:

```text
Dataset
   │
   ▼
Pre-processing / Augmentation
   │
   ▼
Train / Validation / Test Split
   │
   ▼
EfficientNet-B0
   │
   ▼
Multi-Task Training
   │
   ├── Real/Fake Loss
   └── Transformation Loss
   │
   ▼
Model Selection
   │
   ▼
Final Evaluation
```

During training, both objectives are optimized jointly.

Validation performance is used to select the final model configuration before evaluation on the held-out test data.

---

## Key Results

The selected model achieved:

| Metric                                       |                                 Result |
| -------------------------------------------- | -------------------------------------: |
| Real vs AI-generated classification accuracy |                             **91.17%** |
| Architecture                                 |                        EfficientNet-B0 |
| Learning strategy                            |                    Multi-task learning |
| Tasks                                        | Origin + transformation classification |

The multi-task formulation produced a trade-off between the two objectives.

Compared with the corresponding baseline configuration:

* transformation recognition improved by approximately **+0.75 percentage points**;
* real/fake classification decreased by approximately **−1.08 percentage points**.

This suggests that auxiliary transformation supervision can improve the model's awareness of post-processing artifacts, although optimizing both objectives does not automatically improve the primary real/fake classification task.

---

## Main Finding

A key result of the project is that **multi-task learning is not automatically equivalent to higher real/fake detection accuracy**.

Instead, the experiment highlights a more interesting trade-off:

> Learning transformations jointly with image provenance improves the model's ability to reason about post-processing artifacts, while slightly reducing performance on the binary real/fake task.

This indicates that the two objectives are related but not perfectly aligned.

The result is particularly relevant for real-world image-forensics systems, where robustness to transformations may be as important as performance on clean benchmark images.

---

## Why EfficientNet-B0?

EfficientNet-B0 provides a useful balance between:

* model capacity;
* computational efficiency;
* transfer-learning performance;
* parameter count.

This makes it suitable for experimenting with multiple training configurations without requiring an excessively large computational budget.

Its convolutional features also provide a strong baseline for detecting visual and statistical artifacts associated with synthetic images.

---

## Evaluation

Performance is evaluated separately for the two prediction heads.

### Real / AI Classification

Measures the model's ability to distinguish authentic images from AI-generated content.

Relevant metrics include:

* accuracy;
* class-specific performance;
* confusion matrix.

### Transformation Classification

Measures whether the network can identify the post-processing operation associated with an image.

This provides additional information about the type of image statistics learned by the shared backbone.

---

## Robustness Perspective

The project is motivated by a realistic deployment scenario:

```text
AI-generated image
        │
        ▼
Post-processing
        │
        ├── Compression
        ├── Resizing
        ├── Filtering
        └── Other alterations
        │
        ▼
Detection System
```

A detector that only learns artifacts present in pristine generated images may fail when those artifacts are modified.

Jointly learning transformation information is therefore explored as a way of making the representation more aware of these changes.

---

## Tech Stack

* **Python**
* **PyTorch**
* **EfficientNet-B0**
* **Computer Vision**
* **Deep Learning**
* **Transfer Learning**
* **Multi-Task Learning**
* **Image Classification**
* **Image Forensics**

---

## Skills Demonstrated

This project covers several practical deep-learning and computer-vision skills:

* designing multi-head neural-network architectures;
* transfer learning with convolutional backbones;
* multi-task loss formulation;
* dataset preprocessing;
* image classification;
* model training and validation;
* quantitative model comparison;
* robustness analysis;
* experiment design;
* interpretation of conflicting objectives.

---

## Limitations

The project is an experimental study rather than a production-ready AI-content detector.

Important limitations include:

* performance depends on the generators represented in the dataset;
* unseen post-processing pipelines may behave differently;
* robustness to completely unseen image generators is not guaranteed;
* the multi-task objective introduces a trade-off between the two prediction tasks;
* classification accuracy alone does not capture every aspect of forensic reliability.

The reported results should therefore be interpreted within the experimental setup used in the project.

---

## Possible Extensions

Several directions could extend this work:

* evaluate on completely unseen generative models;
* investigate stronger vision backbones;
* compare CNNs with Vision Transformers;
* study frequency-domain features;
* introduce adaptive weighting between the two losses;
* use uncertainty-aware predictions;
* evaluate more aggressive image transformations;
* study cross-dataset generalization;
* investigate contrastive or self-supervised representations.

An especially interesting direction would be dynamically adjusting the relative contribution of the two tasks during training instead of using a fixed multi-task weighting.

---

## Project Context

This project was developed as part of the **Computer Vision** coursework of the MSc in Artificial Intelligence and Robotics at Sapienza University of Rome.

The focus was not only on obtaining high classification accuracy, but also on analyzing the effect of post-processing transformations and the interaction between multiple related learning objectives.

---

## Authors

**Samuele Civale**
MSc Artificial Intelligence and Robotics
Sapienza University of Rome

GitHub: [@samuelecivale](https://github.com/samuelecivale)
