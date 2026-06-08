# Joint Detection of AI-Generated Images and Post-Processing Alterations

This repository contains the official implementation of a **Multi-Task Learning (MTL)** framework designed to simultaneously identify the authenticity of images (Real vs. AI-generated) and track alterations caused by post-processing. This project was developed as part of the **Computer Vision** course.

## 📌 Project Overview
Modern AI-image detectors show significant fragility when images are uploaded to social media, undergoing compression or re-digitization (e.g., photos taken of screens).

Our approach addresses this issue by utilizing Multi-Task Learning with a shared **EfficientNet-B0 backbone** to answer two fundamental forensic questions through a **two-headed architecture**:
1. **Authenticity Head (Binary Classification):** Detects whether the image is Real or AI-generated.
2. **Transformation Head (Multi-class Classification):** Identifies the type of applied alteration (`Original`, `Internet-Transmitted`, or `Re-digitized`).

---

## 📊 Dataset: RRDataset (Balanced Subset)
The model was trained and validated on a balanced subset of the **RRDataset** (Real-world Robustness Dataset), consisting of a total of **6,000 images** distributed as follows:

| Authenticity Class | Original (Clean) | Internet-Transmitted (Compressed) | Re-digitized (Screen-Recapture) |
| :--- | :---: | :---: | :---: |
| **Real** | 1,000 | 1,000 | 1,000 |
| **AI-Generated** | 1,000 | 1,000 | 1,000 |

---

## 🚀 Case Studies Analysis (Experimental Runs)
The project analyzes three different configurations and runs to evaluate absolute accuracy, data stability, and robustness to noisy domains:

### 🔹 Case A: Loss Weight Optimization (Standard MTL - Docs Run)
* **Objective:** Find the optimal balance of the $\alpha$ parameter in the combined Loss function to maximize Real/Fake classification performance.
* **Key Result:** With $\alpha = 0.8$ (80% weight on Real/Fake Loss and 20% on Transformation Loss), the model reaches a peak validation accuracy of **91.25%** on the primary task.
* **Conclusion:** The auxiliary task acts as a regularizer, improving the single-task baseline by +3.61%.

### 🔹 Case B: Stability Across Different Dataset Portions (MTL Alternative Data - Colab Run)
* **Objective:** Validate the framework's consistency by training it on different stratified portions and partitions of the dataset in Colab, checking the model's statistical variance.
* **Key Result:** The model demonstrated excellent stability with a constant mean accuracy of **90.15% ± 0.3%**, proving that the learned features are robust and independent of the specific data split.

### 🔹 Case C: "Artifact-Aware" Architecture with Residual Branch (Colab Run)
* **Objective:** Isolate high-frequency patterns (such as Moiré effects and sensor noise) by introducing a parallel branch that calculates the image residual (`Original Image - Gaussian Blur`).
* **Key Result:** Drastic reduction in the performance drop between clean and degraded images, decreasing from **3.22 percentage points (pp)** in standard models to just **1.66 pp**.
* **Conclusion:** High-frequency feature injection effectively shields the model from re-digitization artifacts.

---

## 📈 Results Comparison Table

| Run Configuration | Experiment Description | Real/Fake Accuracy (Val) | Transformation Acc. | Robustness Drop |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Single-Task Model (Real/Fake only) | 87.64% | - | - |
| **Case A (Docs)** | Standard MTL ($\alpha=0.8$) | **91.25%** | 80.32% | 3.22 pp |
| **Case B (Colab)** | MTL on alternative data portions | 90.15% | 79.44% | 3.10 pp |
| **Case C (Colab)** | MTL with Residual Branch (Artifact-Aware) | 90.10% | 79.52% | **1.66 pp** |

---

## 🛠️ Installation and Usage

1. Clone the GitHub repository:
   ```bash
   git clone [https://github.com/your-username/AI-Detection-Forensics.git](https://github.com/your-username/AI-Detection-Forensics.git)
   cd AI-Detection-Forensics
   2. Open the [Colab Notebook](https://colab.research.google.com/drive/1r4UGEQoh6ycqY3Xq3McsC5gIJpXasoPo) to run the experiments.
3. Ensure the `RRDataset` subset is available in the root folder or linked via drive.

## 📈 Results Summary
| Model | RF Best Accuracy | Transformation Acc | Robustness Drop |
| :--- | :---: | :---: | :---: |
| Baseline (Single Task) | 87.64% | - | - |
| MTL ($\alpha=0.5$) | 90.75% | 84.2% | 3.22 pp |
| MTL ($\alpha=0.8$) | **91.25%** | 80.3% | 3.22 pp |
| **Artifact-Aware** | 90.10% | 79.5% | **1.66 pp** |

## 👥 Contributors
- **Samuele Civale**
- **Alexandru Vivivan Pita**

## 📜 References
- Li et al., "Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection", ICCV 2025.
- Tan & Le, "EfficientNet: Rethinking Model Scaling", ICML 2019.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r4UGEQoh6ycqY3Xq3McsC5gIJpXasoPo#scrollTo=W_YiF-EaMW4_)




