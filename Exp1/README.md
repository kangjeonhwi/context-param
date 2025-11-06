# Exp1: Comparative Analysis of Knowledge Storage and Retrieval in Adapters

## 1. Overview

This `Exp1` experiment aims to analyze the mechanisms by which fine-tuning adapters (e.g., LoRA, DoRA) for Large Language Models (LLMs) store and retrieve specific context (knowledge).

It comprehensively validates the impact of various factors on knowledge storage efficiency, including adapter type, prompt structure, training parameters (epochs), and the relationship between training loss and actual performance.

## 2. Key Research Questions

This experiment aims to address the following key questions:

### 2.1. LoRA vs. DoRA: Comparative Storage Efficiency
* **Question:** Which adapter, LoRA or DoRA, stores contextual (knowledge) information more efficiently?
* **Validation:** Verify if a statistically significant difference exists in knowledge recall performance between the two adapters.

### 2.2. Analysis of Prompt Structure Influence
* **Question:** Does the structure of the prompt used during the knowledge storage (training) phase have a significant impact on final knowledge recall performance?
* **Hypothesis:** If a significant impact is observed, it may suggest that the model has learned a mechanism to retrieve knowledge at the parameter level in response to specific prompts.

### 2.3. Relationship between Epochs and Overfitting
* **Question:** If the number of training epochs is increased to minimize loss and overfit on specific knowledge, does the recall performance for that knowledge improve proportionally?
* **Validation:** If performance improvement is observed, explore and identify the optimal number of epochs to achieve peak performance.

### 2.4. Validation of LoRA Parameter Efficacy
* **Question:** Do the individual parameters within a trained LoRA adapter actually encode meaningful information?
* **Validation:** (Hypothesis) If the parameters store valid information, a significant difference (e.g., variance) should be observable between adapter parameters trained on different contexts. This will be verified through comparative analysis.

### 2.5. Dyprag (Pragmatic-trained LoRA) Performance Comparison
* **Definition:** Dyprag = LoRA trained using a "Pragmatic" method.
* **Question:** Does the "Pragmatic" training method show a significant difference in knowledge storage and retrieval performance compared to the standard LoRA method?
* **Follow-up:** If a significant performance improvement is observed, it will serve as the basis for a follow-up experiment (Exp2).

### 2.6. Correlation between Training Loss and Recall Accuracy
* **Question:** Does knowledge that exhibited high loss during training (i.e., items likely unfamiliar to the model or outside its knowledge boundary) actually show low recall accuracy in both "Naive" and "Prompted" inference settings?
* **Validation:** Analyze the correlation between training loss and final recall accuracy.