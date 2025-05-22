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
import torch

from transformers import AutoModel, AutoTokenizer, AutoConfig

def compute_text_embedding_from_trainer(trainer, dataset, label_type, device):
    model = trainer.model.to(device)
    model.eval()

    encoder = model.deberta

    dataloader = trainer.get_eval_dataloader(dataset) # Create a dataloader so we can handle batches of data

    text_embedding = None
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 1 == 0: # print every 20 batches
                print(f"handling batch {i} out of {len(dataloader)}")
            
            # Move the input to the right device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            # Forward pass
            outputs = encoder(**inputs)
            pooled = model.pooler(outputs.last_hidden_state).detach()  # use the pooler to get the info the classifier needs, i do this because we also train the pooler when transfer learning, so this gives the best accurate task representation
            
            batch_sum = pooled.sum(dim=0)
            batch_size = pooled.size(0)

            # Save the task embedding of this batch, and essentially add all batches together (in the end we divide to get the mean)
            if text_embedding is None:
                text_embedding = batch_sum
            else:
                text_embedding += batch_sum
            count += batch_size
        
        text_embedding /= count # Divide by the amount of batches we handled

        return text_embedding

def parse_args():
    parser = argparse.ArgumentParser(description="Bert variant task embedding")
    parser.add_argument("--dataset_path", type=str, default="Preprocessed_Data/CancerEmo")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--output_dir", type=str, default="./task_embeddings")

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
    print("Setting up the dataset for evaluation")
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
    # Set up the trainer to forward once for the task embedding
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

    # Get the task embedding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embedding = compute_text_embedding_from_trainer(trainer, test_dataset, info["label_type"], device)
    print("Text embedding shape:", text_embedding.shape)
    model_info = parse_checkpoint_string(params.checkpoint_path)
    dataset_name = params.dataset_path.split("/")[-1]

    save_path = f"{params.output_dir}/{dataset_name}/{model_info["path"]}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(text_embedding, f"{save_path}/text_embedding.pt")
    print("Done.")