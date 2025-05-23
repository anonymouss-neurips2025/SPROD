# SPROD: Spurious-Aware Prototype Refinement for Reliable OOD Detection

**Anonymous Authors**  
[NeurIPS 2025 Submission]

---

## Overview

SPROD is a post-hoc Out-of-Distribution (OOD) detection method designed to improve robustness against *unknown spurious correlations*. Unlike most OOD techniques that rely on auxiliary data or require fine-tuning, SPROD is a lightweight, plug-and-play solution compatible with any pretrained feature extractor.

SPROD refines class prototypes through a three-stage process that mitigates spurious biases in representations, enabling superior OOD detection in both binary and multi-class settings, especially under high spurious correlation.

---

## Key Features

- ✅ **Post-hoc & Training-Free**: Works with any frozen backbone — no need for fine-tuning or retraining.
- ✅ **Hyperparameter-Free**: Requires no group annotations or validation data.
- ✅ **Spurious-Aware**: Targets both spurious OOD (SP-OOD) and non-spurious OOD (NSP-OOD) samples.
- ✅ **Generalizable**: Evaluated across 5 diverse and challenging benchmarks.
- ✅ **State-of-the-art**: Outperforms 11 leading post-hoc methods on AUROC and FPR@95.

---

## Method

SPROD refines class prototypes in three stages:
1. **Initial Prototype Construction** – Compute mean feature embeddings for each class.
2. **Classification-Aware Grouping** – Partition samples by correctness and compute subgroup prototypes.
3. **Prototype Refinement** – Reassign training samples to nearest group prototypes to mitigate bias.

For OOD detection, a test sample’s score is its minimum distance to the refined prototypes.

---

## Datasets

We evaluate SPROD on five benchmarks, emphasizing spurious correlations:

| Dataset            | Type        | Key Spurious Attribute       | Classes |
|--------------------|-------------|------------------------------|---------|
| **Waterbirds**     | Synthetic   | Background context           | 2       |
| **CelebA**         | Real-world  | Gender/hair correlation      | 2       |
| **UrbanCars**      | Synthetic   | Background + co-occurring objects | 2   |
| **Spurious ImageNet** | Real-world  | Context-only distractors     | 100     |
| **Animals MetaCoCo** | Real-world  | Backgrounds per animal class | 24      |

---
