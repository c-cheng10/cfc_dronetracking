"""CfC and GRU over event packets, with the event count joined at the readout.

Both see the same per-packet features. The gap (log1p of ms) is always an input
feature; the CfC also uses it inside its gate. A constant-gap arm replaces the
gap with 1 ms everywhere, in the feature and in the gate.
"""

import torch
from torch import nn

from src.models.recurrent import CfCCell


class PacketReadout(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(2 * hidden + 1, 1)

    def forward(self, states, mask, log_count):
        keep = mask.unsqueeze(-1)
        mean = (states * keep).sum(1) / keep.sum(1)
        peak = states.masked_fill(~keep, -torch.inf).amax(1)
        return self.linear(torch.cat((mean, peak, log_count.unsqueeze(1)), dim=1)).squeeze(1)


class PacketCfC(nn.Module):
    def __init__(self, inputs, hidden, backbone, gate_only=False):
        super().__init__()
        self.hidden, self.gate_only = hidden, gate_only
        self.cell = CfCCell(inputs - int(gate_only), hidden, backbone)
        self.readout = PacketReadout(hidden)

    def forward(self, gap, features, mask, log_count):
        # features: [B, T, inputs], column 0 is the gap feature
        x = features[..., 1:] if self.gate_only else features
        h = x.new_zeros(len(x), self.hidden)
        states = []
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h, gap[:, t:t + 1])
            states.append(h)
        return self.readout(torch.stack(states, dim=1), mask, log_count)


class PacketGRU(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.readout = PacketReadout(hidden)

    def forward(self, gap, features, mask, log_count):
        states, _ = self.gru(features)
        return self.readout(states, mask, log_count)


class PacketMixedCfC(nn.Module):
    """CfC with mixed memory: an LSTM cell carries memory and its state enters the CfC.

    Each packet: (h, c) = LSTM(x, (h, c)), then h = CfC(x, h, gap). The LSTM's c
    is copied forward by its own gates, which the plain CfC cell cannot do.
    """

    def __init__(self, inputs, hidden, backbone):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTMCell(inputs, hidden)
        self.cell = CfCCell(inputs, hidden, backbone)
        self.readout = PacketReadout(hidden)

    def forward(self, gap, features, mask, log_count):
        h = features.new_zeros(len(features), self.hidden)
        c = torch.zeros_like(h)
        states = []
        for t in range(features.shape[1]):
            x = features[:, t]
            h, c = self.lstm(x, (h, c))
            h = self.cell(x, h, gap[:, t:t + 1])
            states.append(h)
        return self.readout(torch.stack(states, dim=1), mask, log_count)


class PacketLSTM(nn.Module):
    """Control for PacketMixedCfC: the same memory without the CfC."""

    def __init__(self, inputs, hidden):
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden, batch_first=True)
        self.readout = PacketReadout(hidden)

    def forward(self, gap, features, mask, log_count):
        states, _ = self.lstm(features)
        return self.readout(states, mask, log_count)


# name -> (feature set, constant gap, builder)
PACKET_ARMS = {
    "time_cfc": ("time", False, lambda n: PacketCfC(n, 29, 32)),
    "time_cfc_gate": ("time", False, lambda n: PacketCfC(n, 29, 32, gate_only=True)),
    "time_cfc_const": ("time", True, lambda n: PacketCfC(n, 29, 32)),
    "time_gru": ("time", False, lambda n: PacketGRU(n, 38)),
    "time_gru_const": ("time", True, lambda n: PacketGRU(n, 38)),
    "time_mmcfc": ("time", False, lambda n: PacketMixedCfC(n, 20, 27)),
    "time_mmcfc_const": ("time", True, lambda n: PacketMixedCfC(n, 20, 27)),
    "time_lstm": ("time", False, lambda n: PacketLSTM(n, 33)),
    "space_cfc": ("space", False, lambda n: PacketCfC(n, 28, 32)),
    "space_gru": ("space", False, lambda n: PacketGRU(n, 36)),
}
