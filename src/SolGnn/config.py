from pathlib import Path
from dataclasses import dataclass

def project_root() -> Path:
    """
    This file should live in /SolubilityGNN/src/SolGnn/
    So, we get our project root automatically for convenience
    """
    return Path(__file__).resolve().parents[2]

@dataclass(frozen = True)
class SplitConfig:
    random_state: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

@dataclass(frozen=True)
class TrainConfig:
    device: str = "cpu"
    epochs: int = 200
    batch_size: int = 64

    # early stopping
    patience: int = 50
    min_delta: float = 0.0

    seed: int = 42