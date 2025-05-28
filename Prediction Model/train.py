import argparse
import pickle
from model import FeatureDataset, MLP
import torch
from transformers import set_seed
from transformers import Trainer, TrainingArguments
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier for model-transfer prediction")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data_path", type=str, default="data.pkl")
    parser.add_argument("--model_name", type=str, default="MLP-Classifier")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--train_size", type=float, default=0.8)

    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    params = parse_args()
    set_seed(params.seed)

    #--------
    # Dataset loading
    #--------
    print("loading dataset...")
    generator = torch.Generator().manual_seed(params.seed)
    with open(params.data_path, "rb") as f:
        data = pickle.load(f)
    
    dataset = FeatureDataset(data)
    train_size = int(params.train_size * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator)

    val_size = int(0.1 * train_size)
    final_train_size = train_size - val_size

    train_dataset, val_dataset = random_split(train_dataset, [final_train_size, val_size], generator)

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    #---
    # Setup training
    #---
    # Get input dimensions from the first sample
    x_sample, y_sample = train_dataset[0]
    input_dim = x_sample.shape[0]

    # Initialize model
    model = MLP(input_dim=input_dim, lr=params.lr, hidden_dim=params.hidden_dim)
    logger = TensorBoardLogger("tensorboard_prediction_model_logs", name=params.model_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="checkpoints",
        filename=params.model_name,
        verbose=True
    )
    # #---
    # # Train
    # #---
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = pl.Trainer(
        max_epochs=params.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        accumulate_grad_batches=8
    )

    trainer.fit(model, train_loader, val_loader)




    
