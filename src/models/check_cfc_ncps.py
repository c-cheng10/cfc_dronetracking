"""Check CfCCell against the ncps reference cell. Run: python -m src.models.check_cfc_ncps

ncps computes sigmoid(a*t + b) and mixes ff1*(1-g) + ff2*g; ours computes
sigmoid(-a*t + b) and mixes f1*g + f2*(1-g). So: a -> -a, f1 <-> f2, b unchanged.
Also checks a trained packet model, so the gaps it sees in practice are covered.
"""

import torch
from ncps.torch import CfCCell as NcpsCell

from src.models.recurrent import CfCCell


def copy_into_ncps(ours):
    ref = NcpsCell(ours.backbone.in_features - ours.f1.out_features, ours.f1.out_features,
                   backbone_activation="tanh", backbone_units=ours.backbone.out_features)
    with torch.no_grad():
        ref.backbone[0].load_state_dict(ours.backbone.state_dict())
        ref.ff1.load_state_dict(ours.f2.state_dict())
        ref.ff2.load_state_dict(ours.f1.state_dict())
        ref.time_a.weight.copy_(-ours.a.weight)
        ref.time_a.bias.copy_(-ours.a.bias)
        ref.time_b.load_state_dict(ours.b.state_dict())
    return ref


def worst_difference(ours, steps=200, gap_scale=3.0):
    torch.manual_seed(0)
    ref = copy_into_ncps(ours)
    inputs, hidden = ours.backbone.in_features - ours.f1.out_features, ours.f1.out_features
    h_ours = h_ref = torch.zeros(1, hidden)
    worst = 0.0
    for _ in range(steps):           # batch size 1, as the handoff asked
        x = torch.randn(1, inputs)
        gap = torch.rand(1, 1) * gap_scale
        h_ours = ours(x, h_ours, gap)
        h_ref, _ = ref(x, h_ref, gap)
        worst = max(worst, (h_ours - h_ref).abs().max().item())
    return worst


if __name__ == "__main__":
    from src.models.packet_models import PACKET_ARMS
    random_cell = CfCCell(2, 29, 32)
    _, _, build = PACKET_ARMS["time_cfc"]
    trained = build(2)
    trained.load_state_dict(torch.load("runs/packets-2026-11-22/time_cfc/seed_0.pt", map_location="cpu"))
    for name, cell in (("random weights", random_cell), ("trained time_cfc seed 0", trained.cell)):
        diff = worst_difference(cell)
        print(f"{name}: largest difference over 200 steps {diff:.2e}")
        assert diff < 1e-5, "CfCCell does not match ncps"
    print("CfCCell matches ncps.")
