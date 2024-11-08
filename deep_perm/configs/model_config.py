from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration class"""

    input_size: int
    hidden_sizes: list[int] = None
    dropout_rates: list[float] = None
    # hidden_sizes = [2048, 1024, 512, 256, 128, 64]
    # dropout_rates = [0.2, 0.25, 0.3, 0.35, 0.4, 0.4]
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 20
    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    permeability_threshold: float = 200
    scheduler_type: str = "onecycle"

    # DataIQ parameters
    conf_upper: float = 0.75
    conf_lower: float = 0.25
    aleatoric_percentile: float = 50

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [
                int(self.input_size * 1.2),  # First layer slightly larger than input
                int(self.input_size * 0.8),
                int(self.input_size * 0.4),
                int(self.input_size * 0.2),
                int(self.input_size * 0.05),
                int(self.input_size * 0.01),
            ]
        if self.dropout_rates is None:
            self.dropout_rates = [0.3] * len(self.hidden_sizes)  # Uniform 0.3 dropout
