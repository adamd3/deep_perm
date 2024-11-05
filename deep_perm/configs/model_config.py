from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration class"""

    input_size: int
    hidden_sizes: list[int] = None
    dropout_rates: list[float] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    permeability_threshold: float = 200

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [
                max(self.input_size * 2, 64),
                max(self.input_size, 32),
                max(self.input_size // 2, 16),
                max(self.input_size // 4, 8),
            ]
        if self.dropout_rates is None:
            self.dropout_rates = [0.1, 0.2, 0.3, 0.3, 0.3]
