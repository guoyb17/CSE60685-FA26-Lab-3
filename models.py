"""The completed Lab 2 network; every compact model inherits trained weights."""
import torch
from torch import nn

BASELINE = {'c1': 6, 'c2': 16, 'h1': 120, 'h2': 84}
MODEL_NAMES = ('control', 's25', 's50')
MODEL_CONFIGS = {name: dict(BASELINE, c2=channels) for name, channels in
                 [('source', 16), ('control', 16), ('s25', 12), ('s50', 8)]}


def validate_config(config, name=None):
    if not isinstance(config, dict) or set(config) != set(BASELINE):
        raise ValueError('An architecture needs c1, c2, h1 and h2.')
    if any(type(value) is not int or not 1 <= value <= 1024 for value in config.values()):
        raise ValueError('Architecture widths must be positive integers up to 1024.')
    if name is not None and (name not in MODEL_CONFIGS or config != MODEL_CONFIGS[name]):
        raise ValueError('Architecture does not match the required Lab 3 branch.')
    return dict(config)


class LeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = validate_config(config)
        c1, c2, h1, h2 = (self.config[key] for key in ('c1', 'c2', 'h1', 'h2'))
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Linear(c2 * 4 * 4, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 10))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
