# Joint Detection of AI-Generated Images and Post-Processing Alterations

This repository contains the implementation of a Multi-Task Learning (MTL) framework designed to identify AI-generated images and their corresponding post-processing transformations. This project was developed as part of the **Computer Vision** course.

## 📌 Project Overview
Current AI-image detectors are often fragile when images undergo realistic social-media processing (compression, re-digitization). Our approach addresses this by training a shared **EfficientNet-B0** backbone to simultaneously answer two forensic questions:
1. **Authenticity:** Is the image Real or AI-generated?
2. **Transformation:** What post-processing was applied (Original, Internet-Transmitted, or Re-digitized)?

## 📊 Dataset
We use a balanced subset of the **RRDataset** (Real-world Robustness Dataset).
- **Total Images:** 6,000
- **Structure:** 1,000 images per category (2 Authenticity classes × 3 Transformation classes).
- **Transformations:** 
  - `Original`: High-quality source images.
  - `Internet-transmitted`: Images degraded by social media compression.
  - `Re-digitized`: Images captured from screens (Moiré patterns, sensor noise).

## 🚀 Two-Run Analysis
The project analyzes two distinct experimental configurations ("Runs") to compare absolute accuracy vs. robustness:

### Run 1: Standard MTL Optimization (Reference: Docs)
Focuses on finding the optimal loss weight $\alpha$ for the Real/Fake detection task.
- **Best Alpha:** 0.8 (gives 80% weight to detection, 20% to transformation).
- **Best Accuracy:** **91.25%** on validation set.
- **Outcome:** Proven that the auxiliary task acts as a strong regularizer.

### Run 2: Artifact-Aware Residual Architecture (Reference: Colab)
Introduces a high-frequency residual branch to specifically target re-digitization artifacts.
- **Architecture:** `Image - GaussianBlur` features fused with standard RGB features.
- **Robustness Improvement:** Reduced the accuracy drop from **3.22 pp** (Standard) to **1.66 pp** (Artifact-aware).
- **Outcome:** Significantly more stable performance across noisy domains.

## 🛠️ Installation & Usage
1. Clone the repo:
   ```bash
   git clone [https://github.com/your-username/AI-Detection-Forensics.git](https://github.com/your-username/AI-Detection-Forensics.git)
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

Your professional slide deck on Joint AI-Image Detection and Alteration recognition is ready! I have meticulously integrated the data from both your Colab and Documentation runs to ensure a comprehensive analysis. Let me know if you would like to adjust any of the visual elements or technical details.
# Project_CV
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r4UGEQoh6ycqY3Xq3McsC5gIJpXasoPo#scrollTo=W_YiF-EaMW4_)

[![Google Docs](https://img.shields.io/badge/Google%20Docs-Documento-blue?logo=googledocs&logoColor=white&style=for-the-badge)](https://docs.google.com/document/d/1wLkI6jA8WNPJ3mrpB7pzGrYWUvwfp2-UzjrPwB9O9H8/edit?tab=t.0)


GitHub repository should contains
○ Code
○ Dataset (or the link to it)
○ Project presentation
○ Detailed Readme


Presentation guidelines

● Upload the presentation to GitHub and saved on a USB drive
○ You can use PDF, PowerPoint, or both formats.
● Use white background slides
● Use high color contrast elements for plots and images
● Each presentation should last 10 minutes
● Avoid jumping forward and backward during the presentation
● Ensure to include page numbers on each slide
○ Show the current slide number out of the total slide number

Presentation suggested scheme

● Title: Report the project name, your names, and course information
● Outline: Briefly describe the presentation workflow
● Problem Statement: The challenge you are addressing
● State of the Art: How current research approaches this challenge
● Proposed Method: How you approached this challenge
● Dataset: The data you used for the project
● Experimental Setup: How you configured the elements of the project
● Model Evaluation: How you assessed the performance of your model
● Conclusions: Final considerations and future work
● References: Where you drew inspiration from
