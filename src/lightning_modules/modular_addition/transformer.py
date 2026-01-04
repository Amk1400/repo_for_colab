from dataclasses import asdict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics import Accuracy

from configs.modular_addition.transformer import config
from models.transformer import TransformerClassifier

class TransformerLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters(asdict(config.model))
        self.train_config = config.train
        
        self.model = TransformerClassifier(**asdict(config.model))
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(task="multiclass", num_classes=config.model.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.model.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.model.num_classes)
                
    def forward(self, x, lengths):
        # Generate Causal Mask (Upper triangular -inf)
        seq_len = x.size(1)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        
        return self.model(x, lengths, mask=mask, return_attention=False)

    def training_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        
        logits = self(inputs, lengths)
        
        # Get scalar targets for the last valid position
        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        last_indices = lengths - 1
        final_targets = targets[batch_indices, last_indices]
        
        loss = self.criterion(logits, final_targets)
        
        self.train_acc(logits, final_targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        
        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        last_indices = lengths - 1
        final_targets = targets[batch_indices, last_indices]
        
        self.val_acc(logits, final_targets)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        
        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        last_indices = lengths - 1
        final_targets = targets[batch_indices, last_indices]
        
        self.test_acc(logits, final_targets)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.train_config.optimizer)
        return optimizer