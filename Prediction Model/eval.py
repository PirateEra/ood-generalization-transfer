import argparse
import pickle
import torch
from model import FeatureDataset, MLP
import pytorch_lightning as pl
from transformers import set_seed
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns

def decide_group(q33, q66, prediction):
    if prediction <= q33:
            group = 'Bad'
    elif prediction <= q66:
        group = 'Decent'
    else:
        group = 'Good'
    return group

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Prediction model")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data_path", type=str, default="data.pkl")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--train_size", type=float, default=0.8)
    args, _ = parser.parse_known_args()
    return args

def plot_predictions(true_values, pred_values):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, pred_values, c='blue', label='Samples')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Ideal prediction (y=x)')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('True vs Predicted Regression Scores')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("true_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_pred_groups(true_labels, pred_labels):
    labels = sorted(set(true_labels) | set(pred_labels))  # All unique labels in True or Predicted
    df = pd.DataFrame({'True': true_labels, 'Predicted': pred_labels})
    conf_matrix = pd.crosstab(df['True'], df['Predicted'], normalize='index')

    conf_matrix = conf_matrix.reindex(index=labels, columns=labels, fill_value=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Predictive Model Confusion Matrix')
    plt.savefig("true_vs_predicted_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    params = parse_args()
    set_seed(params.seed)

    print("Loading evaluation dataset...")
    with open(params.data_path, "rb") as f:
        data = pickle.load(f)

    dataset = FeatureDataset(data)
    generator = torch.Generator().manual_seed(params.seed)

    train_size = int(params.train_size * len(dataset))
    test_size = len(dataset) - train_size

    _, test_dataset = random_split(dataset, [train_size, test_size], generator)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    x_sample, y_sample = test_dataset[0]
    input_dim = x_sample.shape[0]

    model = MLP.load_from_checkpoint(params.checkpoint_path, input_dim=input_dim)

    trainer = pl.Trainer(accelerator="auto", devices=1)

    print("Running evaluation...")
    results = trainer.validate(model, dataloaders=test_loader)

    print(f"Evaluation results: {results}")

    # values for these were computed in create_groups_and_data.ipynb
    q33 = 0.166
    q66 = 0.393

    correct = 0
    true_predictions = []
    predictions = []
    true_labels = []
    labels = []
    for i, prediction in enumerate(model.predictions):
        pred_group = decide_group(q33, q66, prediction)
        _, true_y = test_dataset[i]
        true_group = decide_group(q33, q66, true_y.item())
        if pred_group == true_group:
             correct += 1

        true_predictions.append(true_y)
        predictions.append(prediction)
        true_labels.append(true_group)
        labels.append(pred_group)
        print(f"For item {i} we have the following \n groups: true({true_group}) pred({pred_group}) \n predictions: true({true_y}) pred({prediction}\n)")
    
    print(f"The total accuracy is: {correct / len(model.predictions)}")
    plot_predictions(true_predictions, predictions)
    plot_pred_groups(true_labels, labels)
