"""Temporal CNN for classifying annotated event-camera crops."""

import torch
from torch import nn


class EventCountCNN(nn.Module):
    """Score a crop from its log-transformed event counts over time."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 16, kernel_size=9, padding=4),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, log_counts: torch.Tensor) -> torch.Tensor:
        features = self.features(log_counts.unsqueeze(1))
        pooled = torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=1)
        return self.classifier(pooled).squeeze(1)
