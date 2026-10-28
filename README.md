# cfc_dronetracking

Event-camera drone detection with liquid neural networks (CfC), built toward the Canadian IDEaS counter-drone work.

**Core question:** can a small CfC confirm a drone from its rotor flicker in an event stream, and run fast enough for a Raspberry Pi 5? Can it beat (a) a plain FFT frequency baseline and (b) a GRU given Δt, at the same parameter count?

## Layout

```
src/data/      loading event streams, cutting patches, binning events in time
src/models/    FFT baseline, GRU, CfC (ncps first, then our own cell)
src/train/     training and evaluation scripts
experiments/   one markdown log per run: config, seeds, results
notebooks/     exploration and plots
data/          datasets (git-ignored)
```

## Plan

| Step | Goal | Done when |
|---|---|---|
| 0 | Environment + a CfC trains on a toy sine wave | Loss goes to ~0 |
| 1 | Load FRED, plot events around a drone box | You can see the rotors flicker in a plot |
| 2 | Patch dataset: drone vs background, events counted in time bins | Balanced train/val/test split saved to disk |
| 3 | FFT baseline (spectrum + logistic regression) | F1 over 5 seeds |
| 4 | GRU + Δt baseline | F1 over 5 seeds, parameter count recorded |
| 5 | CfC via `ncps` | Same table, same parameter count |
| 6 | Our own CfC cell (batched, variable Δt) | Matches `ncps` accuracy; works at batch size > 1 |
| 7 | Our own DJI Mini 4K recordings; test on SkyEV | Generalization numbers |
| 8 | Latency on a Raspberry Pi 5 | ms per patch |

## Rules

- No single-seed results. Report the mean ± standard deviation over at least 5 seeds.
- Every model compared at the same parameter count.
- Never touch the test split until the final numbers.
- **Bin width ≤ 1 / (4 × rotor blade-pass frequency).** Measure the Mini 4K's frequency with a phone spectrogram first.

## Data

- **FRED** (Florence RGB-Event Drone dataset, arXiv 2506.05163): 7+ h, 5 drone types. One zip per sequence: raw events in `.hdf5`, boxes in `coordinates.txt`. Code: `github.com/miccunifi/FRED`. Put it in `data/FRED/`.
- **SkyEV** (ETH, 2026): held-out generalization test.
