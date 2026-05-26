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


class SmallCNN(nn.Module):
    """Configurable 3-conv-block CNN for CIFAR-10.

    Backbone: 3 × (Conv2d + BatchNorm + ReLU + MaxPool(2)).
    Head (global_pool=False): AdaptiveAvgPool(4) → FC(ch[-1]*16, fc_dim) → FC(fc_dim, n_classes)
    Head (global_pool=True):  AdaptiveAvgPool(1) → FC(ch[-1], n_classes)   ← much smaller

    forward(x) returns raw logits. Use nn.CrossEntropyLoss.
    Compatible with hook-based collect_layer_data() in data_utils.
    """

    def __init__(self, channels=(16, 32, 64), fc_dim=128, n_classes=10, global_pool=False):
        super().__init__()
        in_ch = 3
        blocks = []
        for out_ch in channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        if global_pool:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(channels[-1], n_classes)
        else:
            self.pool = nn.AdaptiveAvgPool2d(4)
            self.classifier = nn.Sequential(
                nn.Linear(channels[-1] * 16, fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fc_dim, n_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x.flatten(1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
