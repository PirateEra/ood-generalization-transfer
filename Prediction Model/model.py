import torch
from torch import nn
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

class MLP(pl.LightningModule):
    def __init__(self, input_dim=2, hidden_dim=64, lr=5e-5, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Make sure we get an output between 0-1 (since we are trying to go for normalised values here)
        )

        self.loss_fn = nn.MSELoss()
        self.val_outputs = []
        self.predictions = []
        
    def forward(self, x):
        return self.model(x).squeeze(-1)
    
    def training_step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.val_outputs.append({'values': preds.detach().cpu().flatten().tolist()})
        self.log('val_loss', loss)
    
    def on_validation_epoch_end(self):
        all_predictions = []

        # Get all batches their outputs
        for output in self.val_outputs:
            all_predictions.extend(output['values'])
    
        self.predictions = all_predictions

        self.val_outputs = [] # Clear for next epoch
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


# The data should be given as follows
# Column features
class FeatureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features']
        label = item['normalized_value']

        # Convert the data to torch tensors to speedup training
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)

        return features, label
    
    # Get the model info this value was from for debug purposes
    def get_info(self, idx):
        item = self.data[idx]
        info = {
            "source_dataset": item['dataset'],
            "target_dataset": item['model'],
            "used_metric": item['metric']
        }
        return info
