# ANLI Multi-Class Classification

## Overview

This project implements an end-to-end machine learning pipeline for the **ANLI (Adversarial Natural Language Inference) Round 2** dataset.

The objective is to classify each premise–hypothesis pair into one of three classes:
- Entailment
- Neutral
- Contradiction

The project focuses on building a **clear, reproducible, and well-structured pipeline**, while evaluating the effectiveness of classical and transformer-based approaches for natural language inference.

---

## Problem Statement

Natural Language Inference (NLI) involves determining the relationship between a premise and a hypothesis.

For each pair, the model predicts whether the hypothesis is:
- supported by the premise (**entailment**)
- unrelated or uncertain (**neutral**)
- contradicted by the premise (**contradiction**)

The ANLI dataset is adversarially constructed, making it particularly challenging and requiring deeper semantic understanding.

---

## Dataset

Source: [Hugging Face ANLI Dataset](https://huggingface.co/datasets/facebook/anli)

This project uses **Round 2**:

- Training set: 45,460 examples  
- Validation set: 1,000 examples  
- Test set: 1,000 examples  

Each sample contains:
- `premise`
- `hypothesis`
- `label`

---

## Pipeline Summary

The project follows a structured machine learning workflow:

1. Data loading and validation  
2. Exploratory Data Analysis (EDA)  
3. Text preprocessing  
4. Baseline model: TF-IDF + Logistic Regression  
5. Transformer models:
   - DistilRoBERTa  
   - RoBERTa  
6. Model evaluation using Accuracy and Macro F1  
7. Comparative analysis and interpretation  

---

## Modeling Approaches

### TF-IDF + Logistic Regression
A classical baseline using sparse lexical features. This provides a reference point for evaluating more advanced models.

### DistilRoBERTa
A lightweight transformer model used to evaluate efficiency-performance tradeoffs.

### RoBERTa
A larger transformer model used to capture deeper contextual relationships.

---

## Configuration

A centralized configuration was used to ensure consistency across transformer experiments.

- max_length = 192  
- learning_rate = 2e-5  
- train_batch_size = 8  
- eval_batch_size = 16  
- gradient_accumulation_steps = 2  
- epochs = 5  
- weight_decay = 0.01  
- warmup_ratio = 0.1  
- lr_scheduler_type = cosine  

This setup balances computational efficiency and model performance while enabling fair comparison across models.

---

## Evaluation Metrics

Models are evaluated using:
- Accuracy  
- Macro F1 Score  

Macro F1 is used to ensure balanced evaluation across all classes.

---

## Results

| Model                         | Accuracy | Macro F1 |
|------------------------------|----------|----------|
| TF-IDF + Logistic Regression | 0.349    | 0.333    |
| DistilRoBERTa                | 0.455    | 0.450    |
| RoBERTa                      | 0.476    | 0.472    |

Best model: RoBERTa achieved the highest test performance with 0.476 accuracy and 0.472 macro F1.

---

## Results Interpretation

Transformer models significantly outperform the TF-IDF baseline, highlighting the importance of contextual representations for NLI tasks.

RoBERTa achieves the best overall performance in both accuracy and macro F1, indicating stronger generalization and class-balanced performance. DistilRoBERTa also performs strongly, demonstrating that lightweight models can deliver meaningful improvements.

### Key Takeaways
- Classical models are insufficient for complex NLI tasks  
- Transformer models provide clear performance gains  
- Larger models improve performance, though with diminishing returns  
- ANLI remains challenging due to its adversarial nature  

---

## Hyperparameter Insights

Experiments were conducted to understand the impact of key hyperparameters:

- Sequence length: Increasing beyond moderate values did not significantly improve performance  
- Epochs: Performance improved early, with diminishing gains in later epochs  
- Learning rate: 2e-5 provided the most consistent results across models  

---

## Pipeline Design

The pipeline is designed to be modular and extensible:

- Hugging Face datasets used for flexible data loading  
- Shared preprocessing across models for consistency  
- Centralized configuration for easy experimentation  
- Reusable training and evaluation components  

This design supports experimentation with:
- different datasets  
- different transformer architectures  
- different hyperparameter configurations  

---

## Reproducibility

The project is structured for reproducibility through:
- centralized parameter configuration  
- consistent preprocessing pipeline  
- clear training and evaluation workflow  

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── Dockerfile
├── notebooks/
│   └── Three_Way_Classification_ANLI.ipynb
├── models/
│   └── anli_r2_roberta/
├── outputs/
│   └── plots/

