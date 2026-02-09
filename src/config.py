from dataclasses import dataclass

RANDOM_SEED = 42

@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

SPLIT = SplitConfig()
