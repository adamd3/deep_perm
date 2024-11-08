import torch.nn as nn
import torch.nn.functional as F


class PermeabilityNet(nn.Module):
    """Neural network for permeability prediction"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        layer_sizes = [config.input_size] + config.hidden_sizes + [2]

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.extend(
                [
                    nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                    nn.BatchNorm1d(layer_sizes[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rates[i]),
                ]
            )

        # Add final output layer separately without dropout
        layers.extend([nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.BatchNorm1d(layer_sizes[-1])])

        self.layers = nn.Sequential(*layers)
        self._init_weights()

    # def __init__(self, config):
    #     super().__init__()
    #     self.config = config

    #     layer_sizes = [
    #         config.input_size,
    #         int(config.input_size * 1.2),  # First layer slightly larger than input
    #         int(config.input_size * 0.8),  # Still wider than input
    #         int(config.input_size * 0.4),  # Start reducing more significantly
    #         int(config.input_size * 0.2),
    #         int(config.input_size * 0.05),
    #         int(config.input_size * 0.01),
    #         2,  # Output layer
    #     ]

    #     layers = []
    #     for i in range(len(layer_sizes) - 1):
    #         layers.extend(
    #             [
    #                 nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
    #                 nn.BatchNorm1d(layer_sizes[i + 1]),
    #                 nn.ReLU(),
    #                 nn.Dropout(0.3),  # Original dropout rate
    #             ]
    #         )

    #     # Remove final ReLU and Dropout after last layer
    #     layers = layers[:-2]

    #     self.layers = nn.Sequential(*layers)
    #     self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass of the neural network"""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.layers(x)

        if self.training:
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)
