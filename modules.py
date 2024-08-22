import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import numpy as np
from torchvision.transforms import v2
import pytorch_lightning as pl
from torchvision.models import efficientnet_v2_m
import torchmetrics
import torch.nn as nn



"""
Create a dataset for the model

"""

class SleepDataset(Dataset):
    # Initialize the dataframe, transformation, and path to mel spectograms
    def __init__(self, df, transforms, mel_dir_path):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.mel_dir_path = Path(mel_dir_path)
        
    def __len__(self):
        return len(self.df)
    
    # Convert the image ot a tensor and label
    def __getitem__(self, idx):
        # Get the row for given index
        row = self.df.iloc[idx]
        # Load the images and convert it into a tensor and get the label
        np_path = self.mel_dir_path / f"{row['file_name']}"
        image = np.load(np_path)
        image = np.float32(image) /  255

        label = torch.tensor(row['label'], dtype=torch.long)
        image = self.transforms(image)
        return image, label
    
"""
Datamodule for model
"""

class SleepDataModule(pl.LightningDataModule):
    def __init__(self, df, split_prop, mel_dir_path, batch_size, num_workers):
        super().__init__()
        self.df = df
        self.split_prop = split_prop
        self.mel_dir_path = mel_dir_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize((512, 512)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        split = int(len(self.df) * self.split_prop)
        self.train_df = self.df[:split]
        self.val_df = self.df[split:]


    def prepare_data(self):
       pass

    
    def setup(self, stage=None):
        #if stage == "fit" or stage is None:
        self.train_dataset = SleepDataset(self.train_df, self.transforms, self.mel_dir_path)
        self.val_dataset = SleepDataset(self.val_df, self.transforms, self.mel_dir_path)

        #if stage == "test" or stage is None:
        #    pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        # Assuming you have a test set to load
        return None

    def predict_dataloader(self):
        # Assuming you have a predict set to load
        return None

        
"""
Model
"""


class SleepModel(pl.LightningModule):
    def __init__(self, learning_rate, num_classes, global_epoch, cam_dir_path):
        super().__init__()
        self.save_hyperparameters()

        default_path = "default_model.pth"

        self.model = efficientnet_v2_m()
        self.model.load_state_dict(torch.load(default_path))

        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        self.lr = learning_rate
        self.num_classes = num_classes
        self.global_epoch = global_epoch
        self.cam_dir_path = cam_dir_path
        
        self.accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

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