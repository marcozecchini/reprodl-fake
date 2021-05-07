#!/usr/bin/env python
# coding: utf-8

# CMD to generate it from jupyter
# jupyter nbconvert --to script "Training.ipynb" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input_prompt=True

import torch, torchaudio
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import logging
logger = logging.getLogger(__name__)

# everytime you want to work with data in Pytorch we need to use Dataset object
class ESC50Dataset(torch.utils.data.Dataset):
    
    # constructor
    def __init__(self, path: Path = Path('data/ESC-50'),
                sample_rate: int = 8000,
                folds = [1]):
        
        # Load CSV file & initialize all torchaudio.transforms
        # Resample --> Melspectrogram --> AmplitudeToDB

        self.path = path
        self.csv = pd.read_csv(self.path /  Path('meta/esc50.csv'))
        self.csv = self.csv[self.csv['fold'].isin(folds)]
        
        self.resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        self.db = torchaudio.transforms.AmplitudeToDB()
        
    #function to index with [] square brackets
    def __getitem__(self, index):
        # Returns (xb, yb) pair
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / 'audio' / row['filename'])
        label = row['target']
        xb = self.db(
            self.melspec(
                self.resample(wav)
            )
        )
        return xb, label
    
    # function tells pytorch how many object you have in this object
    def __len__(self):
        return len(self.csv)

class AudioNet(pl.LightningModule):
 
    def __init__(self, hparams): # or even 16 as base filters without GPU
        super().__init__()
        self.hparams = hparams
        self.conv1 = nn.Conv2d(1, hparams.base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(hparams.base_filters)
        self.conv2 = nn.Conv2d(hparams.base_filters, hparams.base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hparams.base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(hparams.base_filters, hparams.base_filters * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hparams.base_filters * 2)
        self.conv4 = nn.Conv2d(hparams.base_filters * 2, hparams.base_filters * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(hparams.base_filters * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(hparams.base_filters * 4, hparams.n_classes)
 
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])
        return x
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim.lr)
        return optimizer

@hydra.main(config_path="configs", config_name='default')
def train(cfg: DictConfig):

    logger.info(OmegaConf.to_yaml(cfg))

    path = Path(get_original_cwd()) / Path(cfg.data.path)
    train_data = ESC50Dataset(path=path, sample_rate=cfg.data.sample_rate, folds=cfg.data.train_folds)
    val_data = ESC50Dataset(path=path, sample_rate=cfg.data.sample_rate, folds=cfg.data.val_folds)
    test_data = ESC50Dataset(path=path, sample_rate=cfg.data.sample_rate, folds=cfg.data.test_folds)

    # batch size has influence on convergence but also on memory
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)

    # batch size has influence on convergence but also on memory
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size, shuffle=True)

    # batch size has influence on convergence but also on memory
    test_loader =     torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size, shuffle=True)

    pl.seed_everything(0) # always do it
    audionet = AudioNet(cfg.model)
    trainer = pl.Trainer(**cfg.trainer) # Pass each key of the dictionary as a value
    trainer.fit(audionet, train_loader, val_loader)
    trainer.test(audionet, test_loader)


if __name__ == "__main__":
    train()