import argparse
import logging
import os
import shutil
import warnings
from collections import defaultdict

import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore", ".*does not have many workers.*")

import visualization


class MetricsHistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking: return
        t_acc = trainer.callback_metrics.get("train_acc")
        if t_acc is not None: self.history["train_acc"].append(t_acc.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking: return
        v_acc = trainer.callback_metrics.get("val_acc")
        if v_acc is not None: self.history["val_acc"].append(v_acc.item())

def main():
    parser = argparse.ArgumentParser(description="Deep learning on sequential data")
    
    # 1. Task Argument
    parser.add_argument("-t", "--task", type=str, default="pal", 
                        choices=["pal", "palindrome", "mod", "modular_addition"], 
                        help="Task to train on: 'pal' (palindrome) or 'mod' (modular_addition)")
    
    # 2. Model Argument
    parser.add_argument("-m", "--model", type=str, required=True, 
                        choices=["recurrent", "r", "transformer", "t"],
                        help="Model architecture: 'r' (recurrent) or 't' (transformer)")
    
    # 3. Seed Argument with shorthand
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for splitting")
    
    args = parser.parse_args()

    # --- Normalize Arguments ---
    if args.task in ["pal", "palindrome"]:
        task_name = "palindrome"
    elif args.task in ["mod", "modular_addition"]:
        task_name = "modular_addition"
    else:
        raise ValueError("Invalid task type provided.")

    if args.model in ["transformer", "t"]:
        model_type = "transformer"
    elif args.model in ["recurrent", "r"]:
        model_type = "recurrent"
    else:
        raise ValueError("Invalid model type provided.")

    # --- 1. Setup based on model type and task ---
    print(f"Initializing Task: {task_name} | Model: {model_type}")

    if task_name == "palindrome":
        import data.palindrome as data_module
        if model_type == "transformer":
            from configs.palindrome.transformer import config
            from lightning_modules.palindrome.transformer import \
                TransformerLightningModule as ModelClass
        else:
            from configs.palindrome.recurrent import config
            from lightning_modules.palindrome.recurrent import \
                RecurrentLightningModule as ModelClass

    elif task_name == "modular_addition":
        import data.modular_addition as data_module
        if model_type == "transformer":
            from configs.modular_addition.transformer import config
            from lightning_modules.modular_addition.transformer import \
                TransformerLightningModule as ModelClass
        else:
            from configs.modular_addition.recurrent import config
            from lightning_modules.modular_addition.recurrent import \
                RecurrentLightningModule as ModelClass

    # Set Global Seed
    L.seed_everything(args.seed)

    # --- 2. Directory Setup ---
    # Create output directories
    os.makedirs("output/checkpoints", exist_ok=True)
    os.makedirs("output/plots", exist_ok=True)
    
    # Cleanup previous checkpoints for this run
    ckpt_dir_root = os.path.join("output", "checkpoints", task_name, model_type)
    if os.path.exists(ckpt_dir_root):
        shutil.rmtree(ckpt_dir_root)

    # --- 3. Prepare Data ---
    dev_ds, test_ds = data_module.get_base_datasets(seed=args.seed)
    
    test_loader = DataLoader(test_ds, batch_size=config.train.batch_size, 
                             shuffle=False, collate_fn=data_module.collate_fn)

    # K-Fold Splitter
    kfold = KFold(n_splits=config.train.k_folds, shuffle=True, random_state=args.seed)
    
    all_histories = []
    fold_stats = []
    
    best_overall_acc = -1.0
    
    print(f"Starting {config.train.k_folds}-Fold CV for {task_name} - {model_type}...")
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # --- 4. K-Fold Loop ---
    for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(dev_ds)):
        print(f"\n--- Fold {fold_idx+1}/{config.train.k_folds} ---")
        
        if fold_idx > 0:
            logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

        # Subsets and Loaders
        train_sub = Subset(dev_ds, train_ids)
        val_sub = Subset(dev_ds, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=config.train.batch_size, 
                                  shuffle=True, collate_fn=data_module.collate_fn)
        val_loader = DataLoader(val_sub, batch_size=config.train.batch_size, 
                                shuffle=False, collate_fn=data_module.collate_fn)

        # Initialize Module
        model = ModelClass()

        # Callbacks
        history_cb = MetricsHistoryCallback()
        
        # Checkpoint (based on best val accuracy)
        checkpoint_cb = ModelCheckpoint(
            dirpath=os.path.join(ckpt_dir_root, f"fold_{fold_idx}"),
            filename="best-checkpoint",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            verbose=False
        )

        # Trainer
        trainer = L.Trainer(
            max_epochs=config.train.max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[history_cb, checkpoint_cb],
            enable_checkpointing=True, 
            logger=False, 
            enable_progress_bar=False,
            enable_model_summary=(fold_idx == 0)
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # --- Evaluate Best Model ---
        best_ckpt_path = checkpoint_cb.best_model_path
        if best_ckpt_path:
            test_res = trainer.test(ckpt_path=best_ckpt_path, dataloaders=test_loader, verbose=False)
            test_acc = test_res[0]['test_acc']
        else:
            test_acc = 0.0

        # Retrieve stats
        val_acc_hist = history_cb.history['val_acc']
        train_acc_hist = history_cb.history['train_acc']
        
        min_len = min(len(val_acc_hist), len(train_acc_hist))
        if min_len > 0:
            val_acc_hist = val_acc_hist[:min_len]
            train_acc_hist = train_acc_hist[:min_len]
            best_epoch_idx = np.argmax(val_acc_hist)
            best_val_acc = val_acc_hist[best_epoch_idx]
            best_train_acc = train_acc_hist[best_epoch_idx]
        else:
            best_val_acc = 0.0; best_train_acc = 0.0; best_epoch_idx = 0

        fold_stats.append({
            'fold_id': fold_idx,
            'train_acc': best_train_acc,
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'best_epoch': best_epoch_idx 
        })
        
        all_histories.append(history_cb.history)

        if test_acc > best_overall_acc and best_ckpt_path:
            best_overall_acc = test_acc
            checkpoint = torch.load(best_ckpt_path, map_location=lambda s, l: s)
            model.load_state_dict(checkpoint['state_dict'])

    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # --- 5. Generate Plots ---
    print("\nTraining Complete. Generating Plots...")
    
    plot_prefix = os.path.join("output", "plots", f"{task_name}_{model_type}")
    
    visualization.plot_kfold_history(
        all_histories, fold_stats, 
        save_path=f"{plot_prefix}_learning_curve.pdf"
    )
    
    visualization.plot_dumbbell(
        fold_stats, 
        save_path=f"{plot_prefix}_dumbbell.pdf"
    )

    # Cleanup
    if os.path.exists(ckpt_dir_root):
        shutil.rmtree(ckpt_dir_root)

if __name__ == "__main__":
    main()