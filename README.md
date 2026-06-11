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

The balanced subset is split into:

| Split      | Original | Transfer | Redigital | Real Total | AI Total | Total |
| :--------- | -------: | -------: | --------: | ---------: | -------: | ----: |
| Train      |    1,600 |    1,600 |     1,600 |      2,400 |    2,400 | 4,800 |
| Validation |      400 |      400 |       400 |        600 |      600 | 1,200 |

The split is stratified on the joint pair `(real/fake label, transformation label)`, so the validation set preserves the same class balance as the training set.

---

## Task Definition

The model receives one input image and predicts two labels at the same time.

| Task                       | Type                       | Output Classes                |
| :------------------------- | :------------------------- | :---------------------------- |
| Real/Fake Detection        | Binary classification      | Real, AI-generated            |
| Transformation Recognition | Multi-class classification | Original, Transfer, Redigital |

The primary task is **real/fake detection**. The auxiliary task is **transformation recognition**.

The goal is to verify whether transformation supervision can help the model learn more robust forensic features. Instead of only asking whether an image is real or AI-generated, the model also learns whether the image is clean, internet-transmitted, or re-digitized. This is useful because real-world images often contain degradation artifacts that can interfere with AI-generated image detection.

---

## Proposed Method

The proposed architecture is based on **EfficientNet-B0 pretrained on ImageNet**. EfficientNet-B0 was selected because it offers a good trade-off between representation power and computational cost, making it suitable for experiments on Google Colab.

The original EfficientNet classifier is removed and replaced with task-specific heads.

### Single-task baselines

Two single-task baselines are trained:

* **Real/Fake-only baseline:** EfficientNet-B0 with one binary classification head.
* **Transformation-only baseline:** EfficientNet-B0 with one three-class transformation head.

These baselines are used to understand how each task behaves when optimized independently.

### Standard multi-task model

The standard multi-task model uses one shared EfficientNet-B0 backbone and two output heads:

```text
features = EfficientNetB0(image)

out_rf    = Linear(features)   # real/fake: real vs AI-generated
out_trans = Linear(features)   # transformation: original, transfer, redigital
```

The total training loss is computed as a weighted sum of two cross-entropy losses:

```text
L_total = alpha * L_real/fake + (1 - alpha) * L_transformation
```

where:

* `alpha = 0.8` prioritizes real/fake classification while still keeping transformation supervision;
* `alpha = 0.5` gives equal importance to the two tasks;
* `alpha = 0.2` prioritizes transformation recognition;
* the single-task real/fake baseline can be interpreted as the limiting case in which only the real/fake objective is optimized.

The purpose of the ablation is to understand whether the two tasks help each other or compete for representation capacity.

### Artifact-aware variant

An additional **Artifact-Aware Multi-Task EfficientNet** is also evaluated. This model keeps the RGB EfficientNet branch and adds a residual high-frequency branch. The residual branch is obtained by subtracting a blurred version of the image from the original image, making local forensic traces more visible.

This branch is designed to emphasize:

* compression artifacts;
* blur and re-sampling traces;
* redigitalization noise;
* screen recapture artifacts.

The RGB features and artifact features are fused before the two final classification heads.

---

## Experimental Setup

| Component              | Configuration                                      |
| :--------------------- | :------------------------------------------------- |
| Framework              | PyTorch / torchvision                              |
| Backbone               | EfficientNet-B0                                    |
| Pretraining            | ImageNet                                           |
| Input size             | 224 × 224                                          |
| Optimizer              | Adam                                               |
| Learning rate          | 0.0005                                             |
| Weight decay           | 1e-4                                               |
| Batch size             | 32                                                 |
| Epochs                 | 10 for baselines, 10 for multi-task models         |
| Loss function          | Cross-Entropy                                      |
| Main task              | Real/Fake classification                           |
| Auxiliary task         | Transformation classification                       |
| Validation split       | 20%                                                |
| Training set size      | 4,800 images                                       |
| Validation set size    | 1,200 images                                       |
| Seed                   | 42                                                 |
| Mixed precision        | Enabled when CUDA is available                     |

Training data augmentation includes:

* resizing to `224 × 224`;
* random horizontal flip;
* random rotation up to 15 degrees;
* color jitter on brightness and contrast;
* ImageNet normalization.

The validation pipeline is deterministic and only applies resizing, tensor conversion, and ImageNet normalization. This makes validation metrics stable and comparable across models.

The standard multi-task experiments are run with `alpha = 0.2`, `alpha = 0.5`, and `alpha = 0.8`. The artifact-aware model is evaluated with `alpha = 0.5` to allow a fair comparison with the best standard multi-task configuration.

---

## Baseline Results

Two single-task baselines are evaluated.

| Baseline Model               | Accuracy on Real/Fake | Accuracy on Transformation | Notes |
| :--------------------------- | :-------------------: | :------------------------: | :---- |
| Real/Fake-only EfficientNet  |        89.92%         |             -              | Final evaluation of the binary detector |
| Transformation-only EfficientNet |        -          |           86.08%           | Final evaluation of the transformation detector |

During training, the real/fake-only baseline reaches a best validation accuracy of approximately **91.42%**, while the final evaluation reports **89.92%**. The transformation-only baseline reaches **86.08%** in final evaluation, with a best validation accuracy during training of approximately **87.83%**.

The real/fake baseline learns the training distribution very quickly. Its training accuracy reaches approximately **97.5%**, while validation accuracy oscillates around **90-91%**. This train/validation gap suggests overfitting: the model becomes increasingly confident on the training set, but validation performance does not improve consistently after the first epochs.

The transformation-only task is more difficult than binary real/fake detection. Even with a dedicated model, transformation accuracy remains lower than the real/fake accuracy, showing that distinguishing `Original`, `Transfer`, and especially `Redigital` is a harder forensic problem.

---

## Multi-Task Learning Results

The multi-task setup evaluates how the loss weight `alpha` changes the trade-off between the two objectives.

| Alpha | Best Epoch by Mean Accuracy | Main Focus            | Real/Fake Accuracy | Transformation Accuracy | Mean Accuracy | Robustness Drop |
| :---: | :------------------------: | :-------------------- | :----------------: | :---------------------: | :-----------: | :-------------: |
|  0.2  |             7              | Mostly Transformation |       89.17%       |        **84.75%**       |    86.96%     |    -1.00 pp     |
|  0.5  |             4              | Balanced              |     **91.17%**     |         83.42%          |  **87.29%**   |     0.50 pp     |
|  0.8  |             10             | Mostly Real/Fake      |       90.67%       |         81.17%          |    85.92%     |     0.88 pp     |

The best overall configuration is **alpha = 0.5**, which obtains:

* **91.17%** real/fake accuracy;
* **83.42%** transformation accuracy;
* **87.29%** mean accuracy.

Compared with the final real/fake-only baseline accuracy of **89.92%**, the balanced multi-task model improves real/fake detection by:

```text
91.17% - 89.92% = +1.25 percentage points
```

This suggests that transformation recognition can act as a useful auxiliary task. However, the transformation-only baseline still performs better on transformation recognition than the multi-task model: **86.08%** versus **83.42%** for the selected multi-task model. Therefore, multi-task learning is not uniformly better for both tasks, but it is a useful compromise when both forensic questions must be answered by a single model.

---

## Ablation Study Interpretation

The ablation study shows a clear trade-off between the two tasks.

### Alpha = 0.8

This configuration gives more weight to real/fake classification.

It achieves **90.67%** real/fake accuracy, but only **81.17%** transformation accuracy. This confirms that increasing the weight of the primary task helps the binary detector remain strong, but reduces the model's sensitivity to post-processing differences. The mean accuracy is **85.92%**, which is the lowest among the three multi-task configurations.

### Alpha = 0.5

This configuration is the best compromise.

It obtains the highest mean accuracy (**87.29%**) and the best final real/fake accuracy among the multi-task runs (**91.17%**). It also keeps transformation recognition reasonably high (**83.42%**), although it does not beat the transformation-only baseline.

This result suggests that the two tasks provide complementary information when neither of them dominates the training objective.

### Alpha = 0.2

This configuration gives more importance to transformation recognition.

It obtains the best transformation accuracy among the multi-task models (**84.75%**), but real/fake accuracy drops to **89.17%**. This shows that transformation supervision is useful, but giving it too much weight reduces the optimization pressure on the main real/fake task.

---

## Detailed Analysis of the Selected Model

The selected standard multi-task model is the one with **alpha = 0.5**.

| Transformation | Count | Real/Fake Accuracy | Transformation Accuracy |
| :------------- | ----: | :----------------: | :---------------------: |
| Original       |   400 |       91.50%       |         83.75%          |
| Redigital      |   400 |       89.25%       |         77.25%          |
| Transfer       |   400 |       92.75%       |         89.25%          |

`Transfer` is the easiest condition for the selected model, while `Redigital` is the most difficult one. In particular, transformation recognition drops to **77.25%** on redigital images.

The detailed cross-class analysis shows that real redigital images are especially challenging:

| Transformation | Class | Count | Real/Fake Recall | Transformation Accuracy in Group |
| :------------- | :---- | ----: | :--------------: | :------------------------------: |
| Original       | AI    |   200 |      91.50%      |              92.00%              |
| Original       | Real  |   200 |      91.50%      |              75.50%              |
| Redigital      | AI    |   200 |      88.50%      |              85.00%              |
| Redigital      | Real  |   200 |      90.00%      |              69.50%              |
| Transfer       | AI    |   200 |      92.00%      |              94.00%              |
| Transfer       | Real  |   200 |      93.50%      |              84.50%              |

The most fragile group is **real redigital images**, where transformation accuracy falls to **69.50%**. This indicates that re-digitization changes the visual evidence in a way that makes the transformation class harder to recognize, especially for real photographs.

### Confusion Matrices

Real/Fake confusion matrix:

| True / Pred | Predicted Real | Predicted AI |
| :---------- | -------------: | -----------: |
| True Real   |            550 |           50 |
| True AI     |             56 |          544 |

The real/fake detector is relatively balanced: the model makes 50 errors from real to AI and 56 errors from AI to real.

Transformation confusion matrix:

| True / Pred    | Predicted Original | Predicted Transfer | Predicted Redigital |
| :------------- | -----------------: | -----------------: | ------------------: |
| True Original  |                335 |                 59 |                   6 |
| True Transfer  |                 40 |                357 |                   3 |
| True Redigital |                 76 |                 15 |                 309 |

The dominant error is that true redigital images are often predicted as original: **76 redigital images** are classified as original. This confirms that `Redigital` and `Original` share visual patterns that the model does not always separate correctly.

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

| Model                     | Alpha | Real/Fake Accuracy | Transformation Accuracy | Mean Accuracy | Robustness Drop |
| :------------------------ | :---: | :----------------: | :---------------------: | :-----------: | :-------------: |
| Standard Multi-Task       |  0.5  |      **91.17%**    |         83.42%          |  **87.29%**   |     0.50 pp     |
| Artifact-Aware Multi-Task |  0.5  |       90.08%       |       **84.17%**        |    87.13%     |   **-2.00 pp**  |

The artifact-aware model slightly improves transformation recognition, from **83.42%** to **84.17%**, but it reduces real/fake accuracy from **91.17%** to **90.08%**. The mean accuracy remains almost unchanged: **87.13%** compared with **87.29%** for the standard multi-task model.

The robustness drop becomes negative, meaning that the average performance on degraded conditions is not worse than the performance on original images for real/fake detection. This is promising from a robustness perspective, but the model cannot be considered clearly superior overall because it sacrifices some real/fake accuracy.

---

## Final Results Summary

| Model                         | Alpha | Real/Fake Accuracy | Transformation Accuracy | Mean Accuracy | Robustness Drop | Main Takeaway |
| :---------------------------- | :---: | :----------------: | :---------------------: | :-----------: | :-------------: | :------------ |
| RF-only EfficientNet-B0       |   -   |       89.92%       |            -            |       -       |        -        | Strong binary baseline, but prone to overfitting |
| Transformation-only EfficientNet-B0 | - |          -         |         86.08%          |       -       |        -        | Best model when only transformation recognition is required |
| Standard MTL                  |  0.2  |       89.17%       |       **84.75%**        |    86.96%     |    -1.00 pp     | Best transformation accuracy among multi-task runs |
| Standard MTL                  |  0.5  |     **91.17%**     |         83.42%          |  **87.29%**   |     0.50 pp     | Best overall multi-task compromise |
| Standard MTL                  |  0.8  |       90.67%       |         81.17%          |    85.92%     |     0.88 pp     | Strong real/fake performance, weaker transformation recognition |
| Artifact-Aware MTL            |  0.5  |       90.08%       |         84.17%          |    87.13%     |   **-2.00 pp**  | More robust across degraded conditions, but slightly lower real/fake accuracy |

---

## Main Conclusions

The experiments support four main conclusions.

First, **Multi-Task Learning improves real/fake detection** compared with the final RF-only baseline. The selected model with `alpha = 0.5` reaches **91.17%** real/fake accuracy, compared with **89.92%** for the final RF-only baseline.

Second, transformation recognition can work as a form of **auxiliary supervision**. By forcing the backbone to also model whether an image is original, transferred, or re-digitized, the network learns features that are useful for forensic analysis.

Third, the best trade-off depends on the objective:

* `alpha = 0.5` is the best choice for the complete multi-task setup because it obtains the highest mean accuracy and the best final real/fake accuracy;
* `alpha = 0.2` is preferable if transformation recognition is more important;
* `alpha = 0.8` keeps strong real/fake performance but weakens the transformation head;
* the artifact-aware variant is useful when robustness across degraded image conditions is more important than peak real/fake accuracy.

Fourth, **redigital images are the most difficult condition**. The selected model reaches only **77.25%** transformation accuracy on redigital images, and the main confusion is between `Redigital` and `Original`.

Overall, the results suggest that post-processing traces and image history are useful cues for robust AI-generated image detection, but they also introduce a non-trivial trade-off between the two forensic tasks.

---

## Future Work

Possible future improvements include:

* training on a larger subset or on the full RRDataset;
* using stronger cross-validation instead of a single train/validation split;
* evaluating additional backbones such as ResNet, ConvNeXt, ViT, or EfficientNet variants;
* improving the artifact-aware residual branch and testing different high-frequency filters;
* adding macro-F1, ROC-AUC, precision/recall, and confidence calibration metrics;
* adding confidence intervals or statistical significance tests;
* testing generalization on unseen generative models;
* testing generalization on unseen post-processing pipelines;
* adding per-transformation and per-class error analysis to the repository;
* exporting final tables, confusion matrices, and plots automatically as CSV/PNG files;
* making checkpoint loading fully explicit before final evaluation.

---

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Detection-Forensics.git
cd AI-Detection-Forensics
```

Install the required Python packages:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib pillow tqdm
```

Open the Colab notebook:

[Open in Google Colab](https://colab.research.google.com/drive/1pxWcgsUvCyDdLAtXxv-lSceOoOWlS7na#scrollTo=KPeAZNOdaN3s)

Prepare the dataset:

1. Download or prepare the RRDataset archive.
2. Place the file `RRDataset_test.tar.gz` in the following Google Drive folder:

```text
/content/drive/MyDrive/CV/RRDataset_test.tar.gz
```

3. Run the notebook cells in order.

The notebook will:

1. mount Google Drive;
2. extract and inspect the dataset;
3. build a balanced subset of 6,000 images;
4. create stratified train/validation splits;
5. train the single-task baselines;
6. train the multi-task models for different `alpha` values;
7. optionally train the artifact-aware model;
8. compute final metrics, breakdown tables, and confusion matrices.

To run the artifact-aware experiment, set:

```python
RUN_ARTIFACT_AWARE = True
```

in the global configuration section.

---

## Repository Structure

```text
AI-Detection-Forensics/
│
├── README.md
├── CV_project2_colab_fixed_structured.ipynb      # Main Colab implementation notebook
├── CV_project2_analysis_report_IT.pdf            # Detailed Italian analysis report
├── CV_project2_presentation.pptx                 # Final presentation
│
├── checkpoints/                                  # Saved model checkpoints
│   ├── best_single_rf.pth
│   ├── best_single_trans.pth
│   ├── best_standard_multitask_alpha_0.2.pth
│   ├── best_standard_multitask_alpha_0.5.pth
│   ├── best_standard_multitask_alpha_0.8.pth
│   └── best_artifact_aware_alpha_0.5.pth
│
├── outputs/                                      # Generated plots and result tables
│   ├── ablation_results.csv
│   ├── final_comparison.csv
│   ├── confusion_matrix_real_fake.png
│   └── confusion_matrix_transformation.png
│
└── dataset/                                      # Local dataset folder, not committed
    └── RRDataset_test/
```

The dataset and checkpoints should not be committed if they are too large. They can be stored externally or regenerated by running the notebook.

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
