"""CfC and GRU window classifiers with matched parameter counts and readout."""

import torch
from torch import nn


class CfCCell(nn.Module):
    """Closed-form continuous-time cell (Hasani et al., 2022), batched gap per row.

    h' = f1(z) * g + f2(z) * (1 - g), with g = sigmoid(-a(z) * gap + b(z)).
    """

    def __init__(self, inputs, hidden, backbone):
        super().__init__()
        self.backbone = nn.Linear(inputs + hidden, backbone)
        self.f1 = nn.Linear(backbone, hidden)
        self.f2 = nn.Linear(backbone, hidden)
        self.a = nn.Linear(backbone, hidden)
        self.b = nn.Linear(backbone, hidden)

    def forward(self, x, h, gap):
        z = torch.tanh(self.backbone(torch.cat((x, h), dim=1)))
        gate = torch.sigmoid(-self.a(z) * gap + self.b(z))
        return torch.tanh(self.f1(z)) * gate + torch.tanh(self.f2(z)) * (1 - gate)


class Readout(nn.Module):
    """Mean and max over real time steps, as in EventCountCNN."""

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(2 * hidden, 1)

    def forward(self, states, mask):
        keep = mask.unsqueeze(-1)
        mean = (states * keep).sum(1) / keep.sum(1)
        peak = states.masked_fill(~keep, -torch.inf).amax(1)
        return self.linear(torch.cat((mean, peak), dim=1)).squeeze(1)


class EventCfC(nn.Module):
    def __init__(self, hidden=29, backbone=32):
        super().__init__()
        self.hidden = hidden
        self.cell = CfCCell(1, hidden, backbone)
        self.readout = Readout(hidden)

    def forward(self, values, gaps, mask):
        h = values.new_zeros(len(values), self.hidden)
        states = []
        for t in range(values.shape[1]):
            # Padding never reaches the readout, so states past a row's end are harmless.
            h = self.cell(values[:, t:t + 1], h, gaps[:, t:t + 1])
            states.append(h)
        return self.readout(torch.stack(states, dim=1), mask)


class EventGRU(nn.Module):
    """Same tokens, with the gap supplied as a second input feature."""

    def __init__(self, hidden=38):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.readout = Readout(hidden)

    def forward(self, values, gaps, mask):
        states, _ = self.gru(torch.stack((values, torch.log1p(gaps)), dim=-1))
        return self.readout(states, mask)


ARMS = {
    "cfc_dt": (EventCfC, False),
    "cfc_const": (EventCfC, True),
    "gru_dt": (EventGRU, False),
    "gru_const": (EventGRU, True),
}
