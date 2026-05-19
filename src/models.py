import torch.nn as nn


class SimpleMLP(nn.Module):
    """Configurable MLP that returns intermediate feature maps on request."""

    def __init__(self, input_dim=28 * 28, hidden_dims=(20, 10), output_dim=2, activation=nn.Sigmoid):
        super().__init__()
        layers = [nn.Flatten()]
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Softmax(dim=-1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, inference=False):
        feature_maps = []
        for block in self.layers:
            x = block(x)
            if inference:
                feature_maps.append(x)
        return x, feature_maps

    def linear_layer_indices(self):
        return [i for i, m in enumerate(self.layers) if isinstance(m, nn.Linear)]

    def activation_layer_indices(self):
        return [i for i, m in enumerate(self.layers)
                if not isinstance(m, (nn.Flatten, nn.Linear))]
