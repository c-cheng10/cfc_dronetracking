"""Step 0: check that the environment works and a CfC can learn something.

Task: read sin(t), output cos(t). The network needs memory for this,
because one sin value alone doesn't tell you the sign of cos.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from ncps.torch import CfC

torch.manual_seed(0)

# 16 sequences of 100 time steps, each starting at a random phase
t = torch.linspace(0, 4 * np.pi, 100)
phase = torch.rand(16, 1) * 2 * np.pi
x = torch.sin(t + phase).unsqueeze(-1)   # shape (batch, time, features)
y = torch.cos(t + phase).unsqueeze(-1)

model = CfC(1, 32, proj_size=1)          # 1 input, 32 liquid neurons, 1 output
print("params:", sum(p.numel() for p in model.parameters()))
# lr=0.01 makes this CfC collapse to outputting ~0 (loss stuck at 0.5); 0.001 trains fine
opt = torch.optim.Adam(model.parameters(), lr=0.001)

for step in range(500):
    pred, _ = model(x)                   # run the whole sequence through the CfC
    loss = ((pred - y) ** 2).mean()      # mean squared error
    opt.zero_grad()
    loss.backward()                      # backprop through time
    opt.step()
    if step % 50 == 0:
        print(step, round(loss.item(), 4))

plt.plot(y[0, :, 0], label="target cos")
plt.plot(pred[0, :, 0].detach(), label="CfC output")
plt.legend()
plt.title("CfC: sin -> cos")
plt.savefig("experiments/00_sine_cfc.png")
plt.show()
