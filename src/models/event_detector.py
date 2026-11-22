"""A CNN that predicts boxes in full event-camera frames."""

import math

from torch import nn


STRIDE = 4


def conv_block(in_channels, out_channels, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding=dilation,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


class EventDetector(nn.Module):
    """Predict centre scores, subcell offsets and log box sizes on a grid."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, 8, stride=2),
            conv_block(8, 16, stride=2),
            conv_block(16, 24),
            conv_block(24, 24, dilation=2),
            conv_block(24, 24, dilation=4),
        )
        self.centres = nn.Conv2d(24, 1, 1)
        self.geometry = nn.Conv2d(24, 4, 1)
        nn.init.constant_(self.centres.bias, math.log(0.01 / 0.99))

    def forward(self, frames):
        features = self.features(frames)
        return self.centres(features).squeeze(1), self.geometry(features)
