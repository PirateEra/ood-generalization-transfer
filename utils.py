from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def convert_labels(example, labels):
    example["labels"] = [float(bool(example[label])) for label in labels]
    return example

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128) # We can play a bit with this value, doesnt have to be 128

def custom_metric(pred):

    logits = pred.predictions
    labels = pred.label_ids
    probs = 1 / (1 + np.exp(-logits))         # sigmoid
    preds = (probs > 0.5).astype(int)         # threshold at 0.5 so below or above for 0-1

    exact_match = (preds == labels).all(axis=1).mean() 

    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_micro": precision_score(labels, preds, average="micro"),
        "recall_micro": recall_score(labels, preds, average="micro"),
        "subset_accuracy": exact_match
    }