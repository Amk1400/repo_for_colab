from dataclasses import asdict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics import Accuracy

from configs.modular_addition.recurrent import config
from models.recurrent import RecurrentClassifier

class RecurrentLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters(asdict(config.model))
        self.train_config = config.train
        
        self.model = RecurrentClassifier(**asdict(config.model))
        
        # Loss: -100 is used in collate_fn for target padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=config.model.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.model.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.model.num_classes)

    def forward(self, x, lengths):
        # return_last_step_only=False ensures we get (Batch, Seq, Num_Classes)
        # This allows for Deep Supervision (learning the running sum)
        return self.model(x, lengths, return_last_step_only=False)

    def training_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        # logits: (Batch, Seq_Len, Num_Classes)
        logits = self(inputs, lengths)
        
        # Deep Supervision: Flatten to (Batch*Seq, Classes) for CrossEntropy
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        # Accuracy: Only measure on the last valid token of the sequence
        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        last_indices = lengths - 1
        
        final_logits = logits[batch_indices, last_indices]
        final_targets = targets[batch_indices, last_indices]
        
        self.train_acc(final_logits, final_targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        
        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        last_indices = lengths - 1
        
        final_logits = logits[batch_indices, last_indices]
        final_targets = targets[batch_indices, last_indices]
        
        self.val_acc(final_logits, final_targets)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        
        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        last_indices = lengths - 1
        
        final_logits = logits[batch_indices, last_indices]
        final_targets = targets[batch_indices, last_indices]
        
        self.test_acc(final_logits, final_targets)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.train_config.optimizer)
        return optimizer