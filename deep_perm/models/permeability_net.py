import torch.nn as nn
import torch.nn.functional as F


class PermeabilityNet(nn.Module):
    """Neural network for permeability prediction"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build layers dynamically
        layers = []
        in_features = config.input_size

        for i, out_features in enumerate(config.hidden_sizes):
            layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rates[i]),
                ]
            )
            in_features = out_features

        # Output layer
        layers.extend([nn.Linear(in_features, 2), nn.BatchNorm1d(2)])

        self.layers = nn.Sequential(*layers)
        self._init_weights()

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
