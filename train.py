import argparse
from transformers import set_seed
from transformers import Trainer, TrainingArguments
from datasets import Dataset, load_from_disk
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from utils import convert_labels, tokenize_function, custom_metric
from transformers import DataCollatorWithPadding
import json
import os

from transformers import AutoModel, AutoTokenizer, AutoConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Bert variant finetuning")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dataset_path", type=str, default="Preprocessed_Data/CancerEmo")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--freeze_encoder", type=bool, default=False)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--test_size", type=float, default=0.2)

    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    params = parse_args()
    set_seed(params.seed)

    #--------
    # Dataset loading
    #--------
    print("loading dataset...")
    dataset = load_from_disk(params.dataset_path)

    with open(params.dataset_path+"/dataset_info.json") as f:
        info = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer=tokenizer),
        batched=True
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=params.test_size, seed=params.seed)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # To get batches with the right padding

    #--------
    # Training setup
    #--------
    print("loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",
                                                            problem_type="multi_label_classification"
                                                            , num_labels=info["num_labels"])
    # print(model.config)

    if params.freeze_encoder:
        # Freeze all params
        for param in model.deberta.parameters():
            param.requires_grad = False
        # Make sure the classifier is NOT frozen
        for param in model.classifier.parameters():
            param.requires_grad = True
    
        # print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    # The training arguments
    print("loading training args...")
    training_args = TrainingArguments(
        output_dir= params.output_dir,      
        eval_strategy="epoch",      # Evaluate at the end of each epoch
        save_strategy="epoch",
        learning_rate=params.lr,              
        per_device_train_batch_size=params.batch_size,   
        per_device_eval_batch_size=params.batch_size,
        num_train_epochs=params.epochs,               
        weight_decay=params.weight_decay,                # Regularization
        save_total_limit=2,               # Limit checkpoints to save space
        load_best_model_at_end=params.load_model,
        logging_dir=params.logging_dir,             
        logging_steps=100,                
        fp16=params.fp16,                        # Enable mixed precision for faster training
        seed=params.seed
    )

    trainer = Trainer(
        model=model,                        
        args=training_args,             
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,        # Using the proper batching
        compute_metrics=custom_metric
    )


    #--------
    # Training
    #--------
    print("Starting training...")
    trainer.train()




    
