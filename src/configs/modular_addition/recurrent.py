from dataclasses import dataclass
from frozendict import frozendict
from data.modular_addition import VOCAB_SIZE, PAD_TOKEN, MODULUS

@dataclass(frozen=True)
class RecurrentModelConfig:
    vocab_size: int = VOCAB_SIZE
    pad_token: int = PAD_TOKEN
    embed_dim: int = 128      # افزایش ابعاد embedding برای بهتر شدن نمایش نمادها
    hidden_dim: int = 256     # افزایش ظرفیت مخفی برای یادگیری carryها و الگوهای پیچیده‌تر
    num_layers: int = 3       # کمی لایه بیشتر برای افزایش نمایندگی مدل
    num_classes: int = MODULUS
    bidirectional: bool = False
    dropout: float = 0.02     # کاهش dropout (آزمایش: چون آندرفیتینگ داریم، کمتر regularize می‌کنیم)

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64      # کاهش batch برای افزایش به‌روزرسانی‌ها در هر epoch
    max_epochs: int = 20
    k_folds: int = 2
    optimizer: frozendict = frozendict({
        "lr": 1e-3,           # افزایش نرخ یادگیری برای سریع‌تر شدن همگرایی
        "weight_decay": 0.0   # حذف یا کاهش weight decay (برای جلوگیری از underfit)
    })

@dataclass(frozen=True)
class Config:
    model: RecurrentModelConfig = RecurrentModelConfig()
    train: TrainingConfig = TrainingConfig()

config = Config()
