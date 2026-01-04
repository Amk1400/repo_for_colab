# configs/palindrome/transformer.py
from dataclasses import dataclass
from frozendict import frozendict
from data.palindrome import VOCAB_SIZE, PAD_TOKEN

@dataclass(frozen=True)
class TransformerModelConfig:
    vocab_size: int = VOCAB_SIZE
    pad_token: int = PAD_TOKEN
    embed_dim: int = 128   # trade-off: enough capacity but keeps training fast
    mlp_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    num_classes: int = 2
    max_len: int = 600  # must cover extrapolation 500
    dropout: float = 0.1

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 20
    k_folds: int = 2
    optimizer: frozendict = frozendict({
        "lr": 3e-4,
        "weight_decay": 1e-4
    })

@dataclass(frozen=True)
class Config:
    model: TransformerModelConfig = TransformerModelConfig()
    train: TrainingConfig = TrainingConfig()

config = Config()
