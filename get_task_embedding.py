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

def compute_task_embedding_from_trainer(trainer, dataset, label_type, device):
    model = trainer.model.to(device)
    model.eval()

    encoder = model.deberta
    classifier = model.classifier

    # Make sure the encoder has gradient calculations active
    for param in encoder.parameters():
        param.requires_grad = True

    dataloader = trainer.get_eval_dataloader(dataset) # Create a dataloader so we can handle batches of data

    squared_gradients = []

    for _, batch in enumerate(dataloader):
        
        # Move the input to the right device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        labels = batch['labels'].to(device)

        encoder.zero_grad() # Make sure to clear out the gradients before computing new
        classifier.zero_grad()

        # Forward pass
        outputs = encoder(**inputs)
        pooled = model.pooler(outputs.last_hidden_state)  # use the pooler to get the info the classifier needs, i do this because we also train the pooler when transfer learning, so this gives the best accurate task representation
        logits = classifier(pooled) # Get the logits, which we need to compute the loss (which gives us the gradients)

        # This step needs to be dependent on the task, so for multi-label/multi-class/multi-variate u need a different loss function
        if label_type == 'multi-label':
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        elif label_type == 'multi-class':
            loss_fn = torch.nn.functional.cross_entropy
        elif label_type == 'multi-variate':
            loss_fn = torch.nn.functional.mse_loss

        loss = loss_fn(logits, labels)
        loss.backward()

        grads = []
        for param in encoder.encoder.parameters():
            if param.grad is not None:
                grads.append((param.grad.detach() ** 2).flatten()) # Flatten and square the gradients of each parameter

        grads = torch.cat(grads)
        squared_gradients.append(grads) # Add this batch of squared gradients to the array
    
    stacked_gradients = torch.stack(squared_gradients)
    task_embedding = stacked_gradients.mean(dim=0) # .mean() is to take the average over all computed batches

    return task_embedding

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
    task_embedding = compute_task_embedding_from_trainer(trainer, test_dataset, info["label_type"], device)
    print("Task embedding shape:", task_embedding.shape)
    model_info = parse_checkpoint_string(params.checkpoint_path)
    dataset_name = params.dataset_path.split("/")[-1]

    save_path = f"{params.output_dir}/{dataset_name}/{model_info["path"]}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(task_embedding, f"{save_path}/task_embedding.pt")
    print("Done.")