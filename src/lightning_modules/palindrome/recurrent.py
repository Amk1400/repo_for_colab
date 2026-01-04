from dataclasses import asdict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics import Accuracy

from configs.palindrome.recurrent import config
from models.recurrent import RecurrentClassifier

class RecurrentLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Save config for checkpoints
        self.save_hyperparameters(asdict(config.model))
        self.train_config = config.train
        
        # Initialize Model
        self.model = RecurrentClassifier(**asdict(config.model))
        
        # Loss and Metrics
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics - computed per step/epoch automatically
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x, lengths):
        return self.model(x, lengths, return_last_step_only=True)

    # TODO
    def training_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        loss = self.criterion(logits, targets)
        
        self.train_acc(logits, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        
        self.val_acc(logits, targets)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        
        self.test_acc(logits, targets)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        """
        You are free to use any optimizer/scheduler combo you would like.
        Just remember to add the relevant parameters to the config.
        """
        optimizer = optim.AdamW(self.parameters(), **self.train_config.optimizer)
        return optimizer