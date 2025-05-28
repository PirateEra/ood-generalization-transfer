# Emotion Transfer Learning Project

This repository provides scripts and notebooks for training, fine-tuning, evaluating, and analyzing emotion classification models based on DeBERTaV3 and MLP architectures. It includes tools for preprocessing datasets, training models on different emotion datasets, evaluating transfer learning performance, and extracting embeddings for further analysis.

## Installation

To install dependencies using `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ACTS

---
```

## Scripts Overview

### `train.py`

Trains a DeBERTaV3 model on an emotion dataset.

Example command:

```bash
python train.py --dataset_path Preprocessed_Data/CancerEmo
```
| Argument               | Type  | Description                  |
| ---------------------- | ----- | ---------------------------- |
| `--seed`               | int   | Random seed                  |
| `--dataset_path`       | str   | Path to preprocessed dataset |
| `--dynamic_lr`         | bool  | Enable dynamic learning rate (based on dataset size) |
| `--lr`                 | float | Learning rate                |
| `--dynamic_batch_size` | bool  | Enable dynamic batch size (based on dataset size)    |
| `--batch_size`         | int   | Batch size                   |
| `--epochs`             | int   | Number of training epochs    |
| `--weight_decay`       | float | Weight decay                 |
| `--fp16`               | bool  | Mixed-precision training     |
| `--output_dir`         | str   | Output directory             |
| `--logging_dir`        | str   | Logging directory            |
| `--test_size`          | float | Test split ratio             |

---

### `re_train.py`

Fine-tunes the classification head of an existing model checkpoint.

Example command:

```bash
python re_train.py --dataset_path Preprocessed_Data/EmoBank --checkpoint_path results/training/CancerEmo/dataset_CancerEmo_seed_1234_testsize_0.2_bs_16/checkpoint-2096
```
| Argument               | Type  | Description                  |
| ---------------------- | ----- | ---------------------------- |
| `--seed`               | int   | Random seed                  |
| `--checkpoint_path`    | str   | Path to the checkpoint       |
| `--dataset_path`       | str   | Path to preprocessed dataset |
| `--dynamic_lr`         | bool  | Enable dynamic learning rate (based on dataset size)|
| `--lr`                 | float | Learning rate                |
| `--dynamic_batch_size` | bool  | Enable dynamic batch size (based on dataset size)    |
| `--batch_size`         | int   | Batch size                   |
| `--epochs`             | int   | Number of training epochs    |
| `--weight_decay`       | float | Weight decay                 |
| `--fp16`               | bool  | Mixed-precision training     |
| `--output_dir`         | str   | Output directory             |
| `--logging_dir`        | str   | Logging directory            |
| `--test_size`          | float | Test split ratio             |

---

### `eval.py`

Evaluates a model checkpoint on its corresponding dataset.

Example command:

```bash
srun python eval.py --dataset_path Preprocessed_Data/CancerEmo --checkpoint_path results/training/CancerEmo/dataset_CancerEmo_seed_1234_testsize_0.2_bs_16/checkpoint-2096
```
| Argument            | Type | Description                                                   |
| ------------------- | ---- | ------------------------------------------------------------- |
| `--dataset_path`    | str  | Path to preprocessed dataset                                  |
| `--checkpoint_path` | str  | Path to the checkpoint                                        |
| `--directory_path`  | bool | Set to True if evaluating multiple checkpoints in a directory |
| `--output_dir`      | str  | Output directory for evaluation results                       |
---

### `get_task_embedding.py`

Extracts task embeddings from a trained checkpoint.

Example command:

```bash
python get_task_embedding.py --dataset_path Preprocessed_Data/CancerEmo --checkpoint_path results/training/CancerEmo/dataset_CancerEmo_seed_1234_testsize_0.2_bs_16/checkpoint-2096
```
### `get_text_embedding.py`

Extracts text embeddings from a trained checkpoint.

Example command:

```bash
python get_text_embedding.py --dataset_path Preprocessed_Data/CancerEmo --checkpoint_path results/training/CancerEmo/dataset_CancerEmo_seed_1234_testsize_0.2_bs_16/checkpoint-2096
```
---

### `utils.py`

Contains utility functions used for preprocessing and training.

---

## Notebooks

- `project.ipynb`: Initial notebook for setup, training, and dataset exploration.
- `preprocessing.ipynb`: Dataset analysis and preprocessing code for the 10 emotion datasets.
- `evaluation_plotting.ipynb`: Plots and analysis of transfer performance across models.
---

## Folder Structure

- `Preprocessed_Data/`: Contains all the preprocessed datasets used for training and retraining.
- `eval_results/`: Contains all evaluation results for each model and dataset.

---

## Prediction Model (MLP)

Contains logic for training a simple MLP model for emotion prediction.

- `create_groups_and_data.ipynb`: Notebook to preprocess data and define features for the MLP model.
- `train.py`: Script to train the MLP model using the generated `data.pkl` file.
- `eval.py`: Script to evaluate the MLP model and generate performance plots.


