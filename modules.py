import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import numpy as np
from torchvision.transforms import v2
import pytorch_lightning as pl
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
import torchmetrics
import torch.nn as nn



"""
Create a dataset for the model

"""

class SleepDataset(Dataset):
    def __init__(self, df, transforms, mel_dir_path):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.mel_dir_path = Path(mel_dir_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        np_path = self.mel_dir_path / f"{row['file_name']}.npy"
        image_array = np.load(np_path)
        img_array = np.stack((image_array, image_array, image_array), axis=-1)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        image = self.transforms(img_array)
        return image, label
    
"""
Datamodule for model
"""

class SleepDataModule(pl.LightningDataModule):
    def __init__(self, df, split_prop, mel_dir_path, batch_size, num_workers):
        self.df = df
        self.split_prop= split_prop
        self.mel_dir_path = mel_dir_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize((512, 512)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        split = int(len(self.df) * self.split_prop)
        self.train_df = self.df[:split]
        self.val_df = self.df[split:]

    def setup(self, stage = None):
        if stage == "fit" or None:
            self.train_dataset = SleepDataset(self.train_df, self.transforms, self.mel_dir_path)
            self.val_dataset = SleepDataset(self.val_df_df, self.transforms, self.mel_dir_path)

        if stage == 'test' or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return None
    
    def predict_dataloade(self):
        return None

        
"""
Model
"""


class SleepModel(pl.LightningModule):
    def __init__(self, learning_rate, num_classes, global_epoch):
        super().__init__()
        self.save_hyperparameters()

        self.model = efficientnet_v2_m(EfficientNet_V2_M_Weights.DEFAULT)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        self.lr = learning_rate
        self.num_classes = num_classes
        self.global_epoch = global_epoch
        
        self.accuracy_fn = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_hat, y)
        acc = self.accuracy_fn(y_hat, y)
        
        self.log("train_acc", acc, on_step = True, on_epoch=True, logger= True, prog_bar=True)
        self.log("train_loss", loss, on_step = True, on_epoch=True, logger= True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_hat, y)
        acc = self.accuracy_fn(y_hat, y)
       
        self.log("val_acc", acc, on_step = True, on_epoch=True, logger= True, prog_bar=True)
        self.log("val_loss", loss, on_step = True, on_epoch=True, logger= True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.global_epoch += 1
        self.log('global_epoch', self.global_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)