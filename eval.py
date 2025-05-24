import argparse
from transformers import set_seed
from transformers import Trainer, TrainingArguments
from datasets import Dataset, load_from_disk
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from utils import convert_labels, tokenize_function, multi_label_metric, multi_class_metric, multi_variate_metric, parse_checkpoint_string
from transformers import DataCollatorWithPadding
import json
from math import ceil
import os

from transformers import AutoModel, AutoTokenizer, AutoConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Bert variant evaluation")
    parser.add_argument("--dataset_path", type=str, default="Preprocessed_Data/CancerEmo")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--directory_path", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./eval_results")

    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    params = parse_args()
    #----
    # Load the dataset for evaluation and the model from the checkpoint
    #----
    print("Loading the model")
    model_info = parse_checkpoint_string(params.checkpoint_path) # Get the needed info such as the seed from the checkpoint path
    
    set_seed(model_info["seed"])
    # Get the model and tokenizer from the checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(params.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(params.checkpoint_path)
    # Load the dataset
    dataset = load_from_disk(params.dataset_path)

    with open(params.dataset_path+"/dataset_info.json") as f:
        info = json.load(f)

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer=tokenizer),
        batched=True
    )

    # So because we saved the seed, we can simply get the exact same test dataset we used during training
    # Which is the portion that was not used during training
    test_dataset = tokenized_dataset.train_test_split(test_size=model_info["testsize"], seed=model_info["seed"])["test"]
    
    #----
    # Pick the metric method
    #----
    if info["label_type"] == 'multi-label':
        metric = multi_label_metric
    elif info["label_type"] == 'multi-class':
        metric = multi_class_metric
    elif info["label_type"] == 'multi-variate':
        metric = multi_variate_metric

    #----
    # Set up the trainer to evaluate with
    #----
    print("Setting up trainer")
    training_args = TrainingArguments(
        output_dir=params.output_dir,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=model_info["bs"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=metric,  # same one used during training
    )

    # Evaluation
    print("Evaluating")

    results = trainer.evaluate(test_dataset)
    dataset_name = params.dataset_path.split("/")[-1]

    output_dir = params.output_dir+f"/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{model_info["dataset"]}_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(results)
