# GRPO-Summarization: Constrained Text Summarization

This repository contains the implementation of a university project for the **Text Mining** course.  
The goal is to train a model capable of **controlled text summarization**, producing summaries that follow explicit **word or sentence length constraints**.

The project is based on **Grouped Reinforcement Preference Optimization (GRPO)** and **LoRA fine-tuning** using the **Unsloth + vLLM** framework.

---

## Overview

The work is divided into two main stages:

1. **Dataset Creation** – Multiple public summarization datasets are combined and standardized.  
   Long-document datasets are excluded to ensure efficient GRPO training.

2. **Model Training** – A `Llama 3.2 1B Instruct` model is fine-tuned with GRPO, guided by custom reward functions that enforce structure and length adherence.


---

## Repository Contents

- **`create_dataset_final.ipynb`** – Builds and filters the unified summarization dataset.  
- **`GRPO_training_and_evaluation.ipynb`** – Performs GRPO trainingand evaluation.  
- **`Dataset/`** – Contains raw input data, merged datasets, and processed versions ready for GRPO training. 
- **`grpo_saved_lora/`** – Contains the LoRA adapter weights for the trained model.  
- **`per_class_results.csv`** – Summarized evaluation results for each constraint class.

## Replication

All notebooks are **self-contained and guided**, including installation, configuration, and execution steps.  
Running them sequentially fully reproduces the dataset creation, training, and evaluation pipeline.

All training and evaluation runs were **logged using Weights & Biases (W&B)**, ensuring full experiment traceability and reproducibility of results.

**Note:**  
The **Newsroom dataset** is not included in the repository due to file size limits.  
You can download it from:  
[https://lil.nlp.cornell.edu/newsroom/download/index.html](https://lil.nlp.cornell.edu/newsroom/download/index.html)  
Place the test split in the `Dataset/` folder and rename it to:
