from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import re

def convert_labels(example, labels):
    example["labels"] = [float(bool(example[label])) for label in labels]
    return example

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128) # We can play a bit with this value, doesnt have to be 128

def multi_label_metric(pred):

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

def multi_class_metric(pred):

    logits = pred.predictions
    labels = pred.label_ids
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_micro": precision_score(labels, preds, average="micro"),
        "recall_micro": recall_score(labels, preds, average="micro"),
    }

def multi_variate_metric(pred):
    preds = pred.predictions
    labels = pred.label_ids

    return {
        "mse": mean_squared_error(labels, preds),
        "mae": mean_absolute_error(labels, preds),
        "r2": r2_score(labels, preds),
    }


# Fucntion to get the values from a checkpoint location (such as the used seed etc)
def parse_checkpoint_string(path):
    match = re.search(r'dataset_[^/]+', path) # Get the dataset information (starting at dataset_ ending at /)
    if not match:
        return {}

    config_part = match.group()
    parts = config_part.split('_')

    result = {}
    for i in range(0, len(parts) - 1, 2):
        key = parts[i]
        value = try_parse_value(parts[i + 1])
        result[key] = value
    result["path"] = config_part
    return result


def try_parse_value(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value