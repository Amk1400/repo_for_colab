# configs/palindrome/transformer.py
from dataclasses import dataclass
from frozendict import frozendict
from data.palindrome import VOCAB_SIZE, PAD_TOKEN

@dataclass(frozen=True)
class TransformerModelConfig:
    vocab_size: int = VOCAB_SIZE
    pad_token: int = PAD_TOKEN
    embed_dim: int = 96    # ↓ از 128 به 96: محاسبات و حافظه کمتر، هنوز ظرفیت معقول
    mlp_dim: int = 192     # ↓ متناسب با embed_dim
    num_heads: int = 4     # ↓ از 8 به 4: کمتر سربار تقسیمی در headها
    num_layers: int = 3    # ↓ از 6 به 3: نصف شدن لایه‌ها -> سرعت خیلی بهتر
    num_classes: int = 2
    max_len: int = 500     # ↓ از 600 به 300 در train (برای سرعت) — می‌تونی برای تست طول‌های بزرگتر ارزیابی کنی
    dropout: float = 0.1

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128   # ↑ از 32 به 128: کمتر ایترات و بهتر شدن throughput (در صورت حافظه کافی)
    max_epochs: int = 12    # ↓ اپک‌های کمتر؛ با lr بزرگ‌تر سریع‌تر همگرا می‌شه
    k_folds: int = 2
    optimizer: frozendict = frozendict({
        "lr": 5e-4,         # ↑ خفیف تا convergence سریع‌تر
        "weight_decay": 1e-4
    })

@dataclass(frozen=True)
class Config:
    model: TransformerModelConfig = TransformerModelConfig()
    train: TrainingConfig = TrainingConfig()

config = Config()
