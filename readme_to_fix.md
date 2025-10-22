# GRPO-Summarization: Constrained Text Summarization

This repository contains a university project for the **Text Mining** course. The primary objective is to fine-tune a language model to generate high-quality, precise summaries of text documents while adhering to specific constraints, such as a target word or sentence count.

The project involves two main stages:
1.  **Dataset Creation**: Aggregating and preprocessing data from multiple sources to create a unified summarization dataset.
2.  **Model Training**: Fine-tuning a pre-trained language model with LoRA to perform constrained summarization.

---

## Repository Contents

This repository is organized into three main parts: Jupyter Notebooks for the workflow, the dataset files, and the saved model weights.

*   **Jupyter Notebooks**
    *   `create_dataset_final.ipynb`: The starting point of the project. This notebook loads raw data from the `Dataset/` folder, processes it, and merges it into a single file ready for training.
    *   `train_final_tofix.ipynb`: The core of the project. This notebook handles the entire training pipeline, from loading the processed dataset to fine-tuning the model with LoRA and evaluating its performance.

*   **Dataset (`Dataset/` folder)**
    *   This directory contains the source data (`CNN_XSum.json`, `newsroom_test.jsonl`) used to build the final dataset. Notably, `CNN_XSum.json` is a custom-built file not available on public hubs.
    *   It also holds the output of the first notebook, `summary_corpus_merged.json`, which is the direct input for the training process.

*   **Model Weights (`grpo_saved_lora/` folder)**
    *   This directory contains the saved LoRA (Low-Rank Adaptation) adapter weights from our best-performing model. These can be loaded on top of the base model to reproduce the results.

---

## How to Run

The project workflow is divided into two sequential steps, each handled by a dedicated Jupyter Notebook. Both notebooks are self-contained and will install their own dependencies when you run the initial cells.

### Step 1: Create the Dataset

1.  Open and run all cells in the `create_dataset_final.ipynb` notebook.
2.  This will generate the `summary_corpus_merged.json` file in the `Dataset/` directory, which is required for the next step.

### Step 2: Train the Model

1.  Open and run the `train_final_tofix.ipynb` notebook.
2.  This notebook will load the dataset created in the previous step and execute the full training and evaluation pipeline.

---

## Model & Results

### Model Weights

The best-performing model from our experiments was saved using `model.save_lora()`. The resulting adapter weights are stored in the `grpo_saved_lora/` directory.

### Training Logs

All training experiments, including metrics like loss and ROUGE scores, were tracked using **Weights & Biases (W&B)**. Due to the visibility settings of the university account, the W&B run pages are not publicly accessible. However, key results and visualizations are documented within the `train_final_tofix.ipynb` notebook itself.