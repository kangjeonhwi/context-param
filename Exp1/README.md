# 🔬 Exp1: Comparative Analysis of Knowledge Storage and Retrieval in Adapters

## 🚀 Overview

The `Exp1` directory within this repository is dedicated to experiments comparing how effectively different fine-tuning adapters (e.g., LoRA, DoRA) store and retrieve specific contextual knowledge in Large Language Models (LLMs).

The primary objective is to quantitatively and qualitatively evaluate the impact of adapter type, prompt structure, training epochs, and parameter characteristics on knowledge storage and recall performance.

---

## 🧐 Key Research Questions

This set of experiments aims to answer the following core questions:

### 1. LoRA vs. DoRA: Contextual Storage Efficacy
* **Question:** Does LoRA or DoRA more effectively store contextual information (knowledge)?
* **Validation:** Determine if there is a statistically significant difference in knowledge recall performance between the two adapters.

### 2. The Influence of Prompts
* **Question:** Does the prompt structure used during training have a significant impact on knowledge storage?
* **Hypothesis:** If a significant influence is observed, it may suggest that the model has learned to retrieve knowledge at the parameter level to recall specific information.

### 3. Epoch Comparison: Overfitting vs. Performance
* **Question:** If we increase the number of epochs to lower the loss (i.e., overfit on the knowledge), does the performance on related questions improve?
* **Hypothesis:** Improvement is expected (Hypothesis: True). If so, what is the optimal number of epochs for knowledge retention?

### 4. LoRA Parameter Analysis
* **Question:** Do the individual parameters within a trained LoRA adapter actually hold valid, distinct information?
* **Validation:** (Hypothesis) If parameters store meaningful information, the variance or difference between parameters trained on *different* contexts should be significant.

### 5. Dyprag vs. LoRA
* **Definition:** Dyprag = LoRA trained using a "Pragmatic" method.
* **Question:** Is there a significant performance difference in knowledge storage and retrieval when comparing Dyprag to standard LoRA?
* **Follow-up:** If a significant improvement is noted, this will form the basis for Experiment 2 (Exp2).

