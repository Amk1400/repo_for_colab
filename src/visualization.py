import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
COLORS = sns.color_palette("deep")

def plot_kfold_history(histories, fold_stats, save_path="kfold_learning_curve.pdf"):
    """
    Plots raw training and validation accuracy with mean and 1-sigma bands.
    Also plots Test Accuracy markers at the peak validation epoch for each fold,
    with guide lines and annotations.
    
    Args:
        histories: List of dicts containing 'train_acc' and 'val_acc' lists.
        fold_stats: List of dicts containing 'best_epoch' and 'test_acc'.
    """
    print(f"Generating learning curves to {save_path}...")
    
    # Trim to min length if any discrepancy
    min_epochs = min(len(h['train_acc']) for h in histories)
    epochs = np.arange(1, min_epochs + 1)
    
    # Collect arrays: Shape (k, epochs) -> Convert to Percentage
    train_accs = np.array([h['train_acc'][:min_epochs] for h in histories]) * 100
    val_accs = np.array([h['val_acc'][:min_epochs] for h in histories]) * 100
    
    # Compute Raw Stats (No Smoothing)
    train_mean = np.mean(train_accs, axis=0)
    val_mean = np.mean(val_accs, axis=0)
    train_std = np.std(train_accs, axis=0)
    val_std = np.std(val_accs, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plot Mean Lines
    ax.plot(epochs, train_mean, label='Train Mean', color=COLORS[0], linewidth=2)
    ax.plot(epochs, val_mean, label='Val Mean', color=COLORS[1], linewidth=2)
    
    # 2. Shaded Bands
    ax.fill_between(epochs, 
                    train_mean - train_std, 
                    train_mean + train_std, 
                    color=COLORS[0], alpha=0.15, label='Train Std Dev')
    ax.fill_between(epochs, 
                    val_mean - val_std, 
                    val_mean + val_std, 
                    color=COLORS[1], alpha=0.15, label='Val Std Dev')
    
    # 3. Plot Test Accuracy Markers & Annotations
    for i, stat in enumerate(fold_stats):
        # best_epoch is 0-indexed, plot is 1-indexed
        x = stat['best_epoch'] + 1 
        y = stat['test_acc'] * 100
        
        # A. Faint gray dashed line from Y-axis (x=1) to the marker
        ax.hlines(y, 1, x, colors='gray', linestyles='--', linewidth=0.8, alpha=0.5, zorder=3)
        
        # B. Annotation of the Y-value near the Y-axis
        # We place it slightly offset from x=1
        ax.text(1.1, y + 0.5, f"{y:.1f}", color='gray', fontsize=8, fontweight='bold', va='bottom', zorder=4)

        # C. Marker
        label = 'Test Acc (per fold)' if i == 0 else ""
        ax.scatter(x, y, color=COLORS[3], marker='^', s=80, zorder=5, 
                   edgecolors='white', linewidth=0.8, label=label)
    
    # --- Formatting ---
    ax.set_title('K-Fold Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    
    # Dynamic Y-Axis Limits
    global_min = min(np.min(train_accs), np.min(val_accs))
    y_bottom = max(0, global_min - 5)
    ax.set_ylim(y_bottom, 102)
    
    # Y-Axis Ticks
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    
    # X-Axis Limits (Start at 1)
    ax.set_xlim(1, len(epochs) + 1)
    
    # X-Axis Ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    # Grid customization
    ax.grid(True, which='major', linestyle='-', alpha=0.6)
    # Increase visibility of X-axis minor grid lines
    ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=1.0, alpha=0.5)
    ax.yaxis.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_dumbbell(fold_stats, save_path="kfold_dumbbell.pdf"):
    """
    Plots a dumbbell chart comparing Train, Val, and Test accuracy per fold.
    Sorted by Test Accuracy.
    """
    print(f"Generating dumbbell plot to {save_path}...")
    
    # Sort folds by Test Accuracy descending
    fold_stats = sorted(fold_stats, key=lambda x: x['test_acc'], reverse=True)
    
    folds = [f"Fold {s['fold_id']+1}" for s in fold_stats]
    y_pos = np.arange(len(folds))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, stats in enumerate(fold_stats):
        train = stats['train_acc'] * 100
        val = stats['val_acc'] * 100
        test = stats['test_acc'] * 100
        
        # 1. Connect Train to Val (Solid Line)
        ax.plot([train, val], [i, i], color='gray', alpha=0.6, linewidth=2, zorder=1)
        
        # 2. Connect Val to Test (Dashed Line)
        ax.plot([val, test], [i, i], color='gray', alpha=0.6, linestyle='--', linewidth=1.5, zorder=1)
        
        # 3. Plot Points
        ax.scatter(train, i, color=COLORS[0], s=100, zorder=2, 
                   label='Train' if i==0 else "", edgecolors='white')
        ax.scatter(val, i, color=COLORS[1], s=100, zorder=2, 
                   label='Validation' if i==0 else "", edgecolors='white')
        ax.scatter(test, i, color=COLORS[3], s=100, zorder=2, marker='D',
                   label='Test (OOD)' if i==0 else "", edgecolors='white')
        
    # --- Formatting ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(folds, fontsize=11)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Fold Performance (Sorted by Test Acc)", fontsize=14, fontweight='bold')
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.2)
    ax.xaxis.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xlim(-5, 105)
    
    # Invert Y axis so top rank is at top
    ax.invert_yaxis()
    
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()