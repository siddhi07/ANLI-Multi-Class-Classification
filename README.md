# ANLI Multi-Class Classification (Round 2)

## Overview

This project implements an end-to-end machine learning pipeline for the **ANLI (Adversarial Natural Language Inference) Round 2** dataset.

The objective is to classify each premise-hypothesis pair into one of three classes:
- Entailment
- Neutral
- Contradiction

This project was completed as part of a take-home machine learning assignment. The focus was on building a **clear, reproducible, and well-structured pipeline** from exploratory data analysis through model evaluation, while also comparing classical and transformer-based approaches.

---

## Problem Statement

Natural Language Inference (NLI) is the task of determining the relationship between a premise and a hypothesis. For each pair, the model must predict whether the hypothesis is:
- supported by the premise (**entailment**)
- unrelated or uncertain (**neutral**)
- contradicted by the premise (**contradiction**)

The ANLI dataset is especially challenging because it is adversarially constructed, making it harder for models to rely on shallow patterns alone.

---

## Dataset

Source: [Hugging Face ANLI Dataset](https://huggingface.co/datasets/facebook/anli)

This project uses **Round 2** of the dataset:

- **Training set:** 45,460 examples
- **Validation set:** 1,000 examples
- **Test set:** 1,000 examples

Each record contains:
- `premise`
- `hypothesis`
- `label`

---

## Project Goals

The project was designed to satisfy the following requirements:

- Complete end-to-end machine learning pipeline
- Well-documented Jupyter notebook with clear formatting
- Comparison across multiple modeling approaches
- Supporting project documentation
- Docker-ready deployment structure
- Clear presentation of methodology and findings

---

## Pipeline Summary

The notebook follows a structured machine learning workflow:

1. Data loading and validation
2. Exploratory Data Analysis (EDA)
3. Text preprocessing
4. Baseline model: TF-IDF + Logistic Regression
5. Transformer model: DistilRoBERTa
6. Transformer model: RoBERTa
7. Model evaluation using Accuracy and Macro F1
8. Hyperparameter analysis
9. Comparative interpretation of results

---

## Modeling Approaches

### 1. TF-IDF + Logistic Regression
A classical baseline model using sparse lexical features. This provides a simple benchmark for comparison.

### 2. DistilRoBERTa
A lighter transformer model used to evaluate whether a smaller architecture can still provide strong contextual understanding with better efficiency.

### 3. RoBERTa
A larger transformer model used as the strongest contextual baseline in this project.

---

## Configuration

A centralized configuration was used for consistency across transformer experiments.

### Transformer Configuration
- `MODEL_NAME = "roberta-base"`
- `DISTIL_MODEL_NAME = "distilroberta-base"`
- `MAX_LENGTH = 192`
- `LEARNING_RATE = 2e-5`
- `TRAIN_BATCH_SIZE = 8`
- `EVAL_BATCH_SIZE = 16`
- `GRADIENT_ACCUMULATION_STEPS = 2`
- `NUM_EPOCHS = 5`
- `WEIGHT_DECAY = 0.01`
- `WARMUP_RATIO = 0.1`
- `LR_SCHEDULER_TYPE = "cosine"`
- `EARLY_STOPPING_PATIENCE = 2`

This setup was selected to balance runtime, training stability, and model performance while maintaining fair comparison across transformer models.

---

## Evaluation Metrics

The models were evaluated using:

- **Accuracy**
- **Macro F1 Score**

Macro F1 is especially important for this task because it evaluates performance more evenly across all three classes rather than favoring the majority pattern.

---

## Final Results

| Model                         | Accuracy | Macro F1 | Notes                   |
|------------------------------|----------|----------|-------------------------|
| TF-IDF + Logistic Regression | 0.349    | 0.333    | Baseline                |
| DistilRoBERTa                | 0.448    | 0.444    | Lightweight transformer |
| RoBERTa                      | 0.476    | 0.472    | Best-performing model   |

---

## Results Interpretation

Both transformer models significantly outperform the TF-IDF baseline, highlighting the importance of contextual language representations for ANLI.

RoBERTa achieves the best overall performance in both accuracy and macro F1, indicating stronger generalization and better class-balanced performance.

DistilRoBERTa also performs strongly, showing that lightweight transformer models can still deliver meaningful improvements over classical approaches.

### Key Takeaways
- Classical models are insufficient for challenging NLI tasks
- Transformer models provide clear and significant gains
- RoBERTa performs best overall
- DistilRoBERTa offers a strong efficiency-performance tradeoff
- ANLI remains difficult due to its adversarial nature

---

## Hyperparameter Experiments

To better understand model behavior, small-scale experiments were conducted on key hyperparameters.

### Sequence Length
The following values were evaluated:
- 128
- 160
- 256

These experiments showed that increasing sequence length beyond a moderate range did not consistently improve results, while longer sequences increased computational cost. A final value of **192** was used to provide good context coverage while maintaining efficiency.

### Number of Epochs
The following settings were compared:
- 1 epoch
- 2 epochs
- 3 epochs
- 5 epochs

Performance improved across early epochs, with later gains becoming smaller. The final setup used **5 epochs** to allow stronger convergence while remaining computationally practical.

### Learning Rate
The following learning rates were compared:
- `2e-5`
- `1.5e-5`

Across experiments, **2e-5** showed stronger and more consistent performance, so it was selected as the final learning rate.

---

## Practical Note

This notebook was executed using **Google Colab with GPU acceleration**.

To balance runtime and performance, both transformer models were trained with a shared configuration:
- max_length = 192
- learning_rate = 2e-5
- train_batch_size = 8
- eval_batch_size = 16
- gradient_accumulation_steps = 2
- epochs = 5
- weight_decay = 0.01
- warmup_ratio = 0.1
- lr_scheduler_type = cosine

Using a centralized parameter setup ensured consistent and fair comparison across DistilRoBERTa and RoBERTa.

---

## Pipeline Flexibility

The pipeline is designed to be modular and extendable:

- Dataset loading is handled using Hugging Face datasets
- Transformer architectures can be changed by updating model name variables
- Tokenization and training workflows are reusable across models
- Centralized configuration allows easy hyperparameter experimentation

This makes the pipeline suitable for trying:
- different datasets
- different transformer architectures
- different hyperparameter settings

---

## Reproducibility

The notebook is structured for reproducibility through:
- centralized configuration
- deterministic preprocessing steps
- clear training and evaluation workflow
- reusable code blocks across experiments

---

## Project Structure

```text
.
├── notebook.ipynb
├── README.md
├── requirements.txt
├── Dockerfile
├── models/
└── outputs/
