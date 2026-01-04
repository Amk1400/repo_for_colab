# configs/modular_addition/transformer.py
from dataclasses import dataclass
from frozendict import frozendict
from data.modular_addition import VOCAB_SIZE, PAD_TOKEN, MODULUS

@dataclass(frozen=True)
class TransformerModelConfig:
    vocab_size: int = VOCAB_SIZE
    pad_token: int = PAD_TOKEN
    embed_dim: int = 128
    mlp_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    num_classes: int = MODULUS
    max_len: int = 100  # modular addition training lengths up to 20; small buffer for efficiency
    dropout: float = 0.1

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 20
    k_folds: int = 5
    optimizer: frozendict = frozendict({
        "lr": 3e-4,
        "weight_decay": 1e-4
    })

@dataclass(frozen=True)
class Config:
    model: TransformerModelConfig = TransformerModelConfig()
    train: TrainingConfig = TrainingConfig()

config = Config()
