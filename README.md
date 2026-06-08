# Joint Detection of AI-Generated Images and Post-Processing Alterations

This repository contains the implementation of a **Multi-Task Learning framework** for the joint detection of:

1. whether an image is **real or AI-generated**;
2. which **post-processing transformation** has been applied to the image.

The project focuses on real-world scenarios, where images are rarely clean. They are often compressed, transmitted through online platforms, or re-digitized by taking a photo of a screen. These transformations can introduce artifacts that make AI-generated image detection more difficult.

This work was developed for the **Computer Vision Project**.

---

## Project Overview

Most AI-generated image detectors are evaluated under ideal conditions, using clean images directly collected from datasets or generative models. However, in realistic settings, images often undergo transformations such as:

* online transmission or compression;
* re-digitization / screen recapture;
* post-processing operations that alter visual artifacts.

The main idea of this project is that learning the **history of an image** can help the model become more robust.

For this reason, we use a **Multi-Task EfficientNet-B0** architecture with a shared backbone and two classification heads:

* **Real/Fake Head:** binary classification between real and AI-generated images.
* **Transformation Head:** multi-class classification among `Original`, `Transfer`, and `Redigital`.

---

## Dataset: RRDataset Balanced Subset

The experiments are based on a balanced subset of the **RRDataset**, a real-world robustness benchmark containing both real photographs and AI-generated images.

The subset contains **6,000 images**, distributed across two authenticity classes and three transformation categories.

| Authenticity Class |  Original | Transfer / Internet-Transmitted | Redigital / Re-digitized |   Total   |
| :----------------- | :-------: | :-----------------------------: | :----------------------: | :-------: |
| Real               |   1,000   |              1,000              |           1,000          |   3,000   |
| AI-Generated       |   1,000   |              1,000              |           1,000          |   3,000   |
| **Total**          | **2,000** |            **2,000**            |         **2,000**        | **6,000** |

The transformation labels are defined as follows:

| Label | Transformation | Description                                         |
| :---: | :------------- | :-------------------------------------------------- |
|   0   | Original       | Clean image without additional post-processing      |
|   1   | Transfer       | Image transmitted online or affected by compression |
|   2   | Redigital      | Image re-captured from a screen or similar process  |

---

## Task Definition

The model receives one input image and predicts two labels at the same time.

| Task                       | Type                       | Output Classes                |
| :------------------------- | :------------------------- | :---------------------------- |
| Real/Fake Detection        | Binary classification      | Real, AI-generated            |
| Transformation Recognition | Multi-class classification | Original, Transfer, Redigital |

The goal is to verify whether the auxiliary transformation task can improve the robustness of the primary real/fake detection task.

---

## Proposed Method

The proposed architecture is based on **EfficientNet-B0 pretrained on ImageNet**.

The original classifier is removed and replaced with two independent heads:

* a **2-class head** for real/fake prediction;
* a **3-class head** for transformation prediction.

The total training loss is computed as:

```text
L_total = alpha * L_real/fake + (1 - alpha) * L_transformation
```

where:

* `alpha = 1.0` corresponds to the single-task baseline;
* `alpha = 0.8` prioritizes real/fake classification while keeping transformation supervision;
* `alpha = 0.5` gives equal importance to both tasks;
* `alpha = 0.2` prioritizes transformation recognition.

---

## Experimental Setup

| Component      | Configuration                 |
| :------------- | :---------------------------- |
| Framework      | PyTorch / torchvision         |
| Backbone       | EfficientNet-B0               |
| Pretraining    | ImageNet                      |
| Input size     | 224 × 224                     |
| Optimizer      | Adam                          |
| Learning rate  | 0.0005                        |
| Weight decay   | 1e-4                          |
| Batch size     | 32                            |
| Epochs         | 10                            |
| Loss function  | Cross-Entropy for both heads  |
| Main task      | Real/Fake classification      |
| Auxiliary task | Transformation classification |

Data augmentation includes:

* resizing;
* random horizontal flip;
* random rotation;
* color jitter;
* normalization.

The baseline is trained only for the real/fake task, while the multi-task models are trained using both the real/fake and transformation labels.

---

## Baseline Results

The baseline is a single-task EfficientNet-B0 trained only for real/fake classification.

| Metric                    |              Value             |
| :------------------------ | :----------------------------: |
| Best validation accuracy  |             87.64%             |
| Final validation accuracy |             86.27%             |
| Final training accuracy   |             97.50%             |
| Training / validation gap | Around 11–12 percentage points |

The baseline reaches high training accuracy but its validation accuracy plateaus early and oscillates. This suggests that the model learns the training distribution very well, but struggles to generalize.

The main issue is overfitting: the model becomes very confident on the training set, while validation performance remains almost unchanged after the first epochs.

---

## Multi-Task Learning Results

The multi-task setup improves the real/fake classification accuracy in all tested configurations.

| Alpha | Main Focus            | Best Real/Fake Accuracy | Best Transformation Accuracy | Interpretation                                              |
| :---: | :-------------------- | :---------------------: | :--------------------------: | :---------------------------------------------------------- |
|  1.0  | Real/Fake only        |          87.64%         |               -              | Baseline; unstable and prone to overfitting                 |
|  0.8  | Mostly Real/Fake      |        **91.25%**       |            80.33%            | Best real/fake performance                                  |
|  0.5  | Balanced              |          90.75%         |            84.25%            | Most balanced and stable configuration                      |
|  0.2  | Mostly Transformation |          89.33%         |          **84.83%**          | Strong transformation recognition, lower real/fake priority |

The best real/fake result is obtained with `alpha = 0.8`, reaching **91.25% validation accuracy**.

Compared to the baseline, this corresponds to an improvement of:

```text
91.25% - 87.64% = +3.61 percentage points
```

---

## Ablation Study Interpretation

The ablation study shows a clear trade-off between the two tasks.

### Alpha = 0.8

This configuration gives the best performance on the main task.

The model gives more importance to real/fake classification, while the transformation task still acts as a regularizer. This helps reduce overfitting without distracting the model too much from the primary objective.

### Alpha = 0.5

This configuration is the most balanced.

It achieves slightly lower real/fake accuracy than `alpha = 0.8`, but it improves transformation recognition and shows more stable validation behavior.

### Alpha = 0.2

This configuration gives more importance to the transformation task.

It obtains the best transformation accuracy, but the real/fake task receives less optimization pressure. As a result, real/fake performance is still better than the baseline, but lower than the other multi-task configurations.

---

## Artifact-Aware Variant

An additional experiment introduces an **Artifact-Aware Multi-Task EfficientNet**.

This variant adds a residual / high-frequency branch designed to emphasize forensic traces such as:

* compression artifacts;
* blur;
* re-sampling traces;
* redigitalization noise;
* screen recapture artifacts.

The goal is not only to maximize overall accuracy, but also to reduce the performance gap between clean and degraded images.

| Model                     | Real/Fake Accuracy | Transformation Accuracy | Robustness Drop |
| :------------------------ | :----------------: | :---------------------: | :-------------: |
| Standard Multi-Task       |       90.15%       |          79.44%         |     3.22 pp     |
| Artifact-Aware Multi-Task |       90.10%       |          79.52%         |   **1.66 pp**   |

Although the artifact-aware model does not significantly improve overall real/fake accuracy, it reduces the robustness drop from **3.22 pp** to **1.66 pp**.

This suggests that explicitly modeling high-frequency artifacts can help make the model more consistent across original and degraded images.

---

## Final Results Summary

| Model                    | Real/Fake Accuracy | Transformation Accuracy | Robustness Drop | Main Takeaway                 |
| :----------------------- | :----------------: | :---------------------: | :-------------: | :---------------------------- |
| Baseline EfficientNet-B0 |       87.64%       |            -            |        -        | Learns real/fake but overfits |
| MTL, alpha = 0.8         |     **91.25%**     |          80.33%         |     3.22 pp     | Best real/fake accuracy       |
| MTL, alpha = 0.5         |       90.75%       |          84.25%         |     3.22 pp     | Best balance between tasks    |
| MTL, alpha = 0.2         |       89.33%       |        **84.83%**       |        -        | Best transformation accuracy  |
| Artifact-Aware MTL       |       90.10%       |          79.52%         |   **1.66 pp**   | Best robustness consistency   |

---

## Main Conclusions

The experiments support three main conclusions.

First, **Multi-Task Learning improves real/fake detection** compared to the single-task baseline. All tested multi-task configurations outperform the baseline.

Second, transformation recognition works as a form of **regularization**. By forcing the model to also understand whether an image is original, transferred, or re-digitized, the shared backbone learns more useful and robust visual representations.

Third, the best trade-off depends on the objective:

* `alpha = 0.8` is the best choice if the goal is maximum real/fake accuracy;
* `alpha = 0.5` is the best choice if the goal is a balanced and stable model;
* the artifact-aware variant is useful when robustness across degraded image conditions is more important than peak accuracy.

Overall, the results suggest that post-processing traces and image history are useful cues for robust AI-generated image detection.

---

## Future Work

Possible future improvements include:

* training on a larger subset or on the full RRDataset;
* using stronger cross-validation;
* evaluating additional backbones and architectures;
* improving the artifact-aware residual branch;
* adding confusion matrices and per-transformation error analysis;
* testing generalization on unseen generative models;
* testing generalization on unseen post-processing pipelines.

---

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Detection-Forensics.git
cd AI-Detection-Forensics
```

Open the Colab notebook:

[Open in Google Colab](https://colab.research.google.com/drive/1r4UGEQoh6ycqY3Xq3McsC5gIJpXasoPo)

Make sure that the RRDataset subset is available in the project root folder or mounted through Google Drive.

---

## Repository Structure

```text
AI-Detection-Forensics/
│
├── CV_project2.ipynb          # Main implementation notebook
├── CV_Project.pdf             # Result analysis and interpretation
├── README.md                  # Project documentation
└── [dataset folder]           # RRDataset subset, if locally available
```

---

## Contributors

* Samuele Civale
* Alexandru Vivian Pita

---

## References

* Li et al., “Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios”, ICCV 2025.
* Shao et al., “Detecting and Grounding Multi-modal Media Manipulation”, CVPR 2023.
* Tan and Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks”, ICML 2019.

---

## Export / Submission Checklist

Before submission:

* [ ] Export the presentation as PowerPoint.
* [ ] Export the presentation as PDF.
* [ ] Upload the repository and presentation to GitHub.
* [ ] Copy the presentation to a USB drive.
* [ ] Verify that all plots and tables are readable on a white background.
* [ ] Check that all reported results match the notebook and analysis document.

