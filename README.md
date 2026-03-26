# ANLI-Multi-Class-Classification

## Overview
This project implements an end-to-end machine learning pipeline for the **ANLI (Adversarial Natural Language Inference) Round 2** dataset.

The objective is to classify each premise-hypothesis pair into one of three classes:
- Entailment
- Neutral
- Contradiction

This project focuses on building a **clear, reproducible, and well-structured ML pipeline**, while also analyzing tradeoffs between model complexity, performance, and efficiency.

---

## Project Pipeline

The workflow follows a structured approach:

1. Data loading and validation  
2. Exploratory Data Analysis (EDA)  
3. Baseline model (TF-IDF + Logistic Regression)  
4. Lightweight transformer (DistilRoBERTa)  
5. Full transformer model (RoBERTa)  
6. Model comparison and evaluation  
7. Error analysis and insights  

---

## Dataset

- Source: Hugging Face `facebook/anli`
- Split used:
  - `train_r2`
  - `dev_r2`
  - `test_r2`

ANLI is an **adversarial dataset**, designed to challenge models with:
- subtle reasoning
- ambiguous phrasing
- complex semantic relationships

---

## Models Used

### 1. Baseline: TF-IDF + Logistic Regression
- Classical machine learning approach
- Provides a reference performance level

### 2. DistilRoBERTa
- Lightweight transformer model
- Faster and more efficient
- Demonstrates performance gain over baseline

### 3. RoBERTa (Final Model)
- Strong transformer model
- Captures deeper semantic relationships
- Achieves best performance

---

## Results

| Model | Accuracy | Macro F1 |
|------|----------|----------|
| TF-IDF + Logistic Regression | 0.348 | 0.3324 |
| DistilRoBERTa | 0.403 | 0.3926 |
| RoBERTa | **0.414** | **0.4052** |

---

## Key Insights

- Transformer models significantly outperform classical baselines  
- DistilRoBERTa provides a strong efficiency-performance tradeoff  
- RoBERTa achieves the best results but with higher computational cost  
- ANLI remains a challenging dataset, with moderate accuracy even for strong models  

---

## Sequence Length Analysis

Text length analysis showed that most samples fall within 100 words.

Experiments with different sequence lengths:
- `max_length = 128` → best performance  
- `160` and `256` → no improvement  

Final choice:
MAX_LENGTH = 128

---

## Hyperparameter Insights

### Sequence Length
- Most samples fall within 100 words  
- `max_length = 128` captures sufficient information  
- Increasing to 160 or 256 did not improve performance  

### Learning Rate
- `2e-5` consistently outperformed `1.5e-5`  
- Higher learning rate enabled better convergence within limited epochs  

### Training Epochs
- Performance improved from 1 → 3 epochs  
- Gains beyond this were modest relative to computation cost  

---

## Error Analysis

- Most confusion occurs between **neutral and contradiction**  
- Errors often involve implicit reasoning or ambiguous phrasing  
- Entailment cases are easier due to more explicit relationships  

This highlights the difficulty of reasoning-based classification in adversarial datasets.

---


