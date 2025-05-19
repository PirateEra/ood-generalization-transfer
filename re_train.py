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
    parser = argparse.ArgumentParser(description="Bert variant finetuning")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--dataset_path", type=str, default="Preprocessed_Data/CancerEmo")
    parser.add_argument("--dynamic_lr", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dynamic_batch_size", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./retrained_results")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--test_size", type=float, default=0.2)

    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    params = parse_args()
    set_seed(params.seed)

    #--------
    # Dataset loading, make sure your data is preprocessed (for examples of the right format see preprocessing.ipynb)
    #--------
    print("loading dataset...")
    dataset = load_from_disk(params.dataset_path)

    with open(params.dataset_path+"/dataset_info.json") as f:
        info = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(params.checkpoint_path)
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer=tokenizer),
        batched=True
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=params.test_size, seed=params.seed)
    val_size = 0.1  # use 10% of the train dataset that is leftover, for validation between epochs
    train_val = split_dataset['train'].train_test_split(test_size=val_size, seed=params.seed)
    #
    train_dataset = train_val['train']
    val_dataset = train_val['test']
    test_dataset = split_dataset['test']

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # To get batches with the right padding

    #--------
    # Training setup
    #--------
    if info["label_type"] == 'multi-label':
        problem_type = "multi_label_classification"
        metric = multi_label_metric
        best_metric = "eval_f1_micro"
        greater_is_better = True
    elif info["label_type"] == 'multi-class':
        problem_type = "single_label_classification"
        metric = multi_class_metric
        best_metric = "eval_f1_macro" # Used f1 instead of accuracy due to class imbalance example for XED = {5: 745, 0: 425, 4: 300, 2: 225, 6: 178, 1: 161, 3: 96}
        greater_is_better = True
    elif info["label_type"] == 'multi-variate':
        problem_type = "regression"
        metric = multi_variate_metric
        best_metric = "eval_loss"
        greater_is_better = False

    #--
    # Training parameters (dynamic)
    #--
    print("Setting the training parameters...")
    size = len(split_dataset["train"])

    if params.dynamic_batch_size:
        params.batch_size = 8 if size < 5000 else 16 if size < 10000 else 32

    if params.dynamic_lr:
        params.lr = 1e-5 if size < 5000 else 2e-5 if size < 10000 else 3e-5

    steps_per_epoch = ceil(size / params.batch_size)
    log_times_per_epoch = 5
    logging_steps = max(1, steps_per_epoch // log_times_per_epoch) # This implies we log x times per epoch

    print(f"loading checkpoint with problem type {problem_type}...")
    # load the encoder from the checkpoint
    encoder = AutoModel.from_pretrained(params.checkpoint_path)

    # Load deberta v3 (for a new classifier)
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=info["num_labels"],
        problem_type=problem_type,
    )

    # Replace the encoder of the new model with the trained (on a different task)
    model.deberta = encoder

    # Freeze encoder
    for param in model.deberta.parameters():
        param.requires_grad = False

    # Unfreeze classifier to make sure it can be trained
    for param in model.classifier.parameters():
        param.requires_grad = True

    
        # print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    # The training arguments
    print("loading training args...")
    dataset_name = params.dataset_path.split("/")[-1]
    training_args = TrainingArguments(
        output_dir= f"{params.output_dir}/retraining/{dataset_name}/dataset_{dataset_name}_seed_{params.seed}_testsize_{params.test_size}_bs_{params.batch_size}",      
        eval_strategy="epoch",      # Evaluate at the end of each epoch
        save_strategy="epoch",
        learning_rate=params.lr,              
        per_device_train_batch_size=params.batch_size,   
        per_device_eval_batch_size=params.batch_size,
        num_train_epochs=params.epochs,               
        weight_decay=params.weight_decay,                # Regularization, prevent the model from learning too large weights
        save_total_limit=2,               # Limit checkpoints to save space
        load_best_model_at_end=True,
        logging_dir=f"{params.logging_dir}/{dataset_name}/dataset_{dataset_name}_seed_{params.seed}_testsize_{params.test_size}_bs_{params.batch_size}",             
        logging_steps=logging_steps,            # dynamically log 5 times per epoch    
        fp16=params.fp16,                        # Enable mixed precision for faster training
        seed=params.seed,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,                        # First 10% the LR gradually increases from 0 to specified LR
        greater_is_better=greater_is_better,     # Based on the better metric, for loss for example greater is not better
        metric_for_best_model = best_metric, # This is a metric that decides what model performed the best during each epoch, based on the task
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,                        
        args=training_args,             
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,        # Using the proper batching
        compute_metrics=metric,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], # if after 2 epochs no improvement is made, earlystop
    )

    #--------
    # Training
    #--------
    print("Starting training...")
    trainer.train()




    
