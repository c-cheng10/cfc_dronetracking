# CfC handoff

Written November 22, 2026, on the `max` branch. The question, task and protocol are in [CFC_EXPERIMENT.md](CFC_EXPERIMENT.md); every result and its caveats are in [FINDINGS.md](FINDINGS.md), newest first.

## Where it stands

A CfC does not yet beat a GRU of the same size on FRED drone windows. With identical event packets, the GRU scores 0.979 ± 0.001 AUC and 0.911 ± 0.006 matched AUC on validation, the CfC 0.960 ± 0.011 and 0.851 ± 0.006. The CfC's matched AUC is level with the CNN's (0.853). "Matched AUC" compares windows whose event counts are nearly equal, where event count alone scores 0.502.

Timing does help the CfC: real packet gaps beat constant gaps by 0.066 matched AUC in 5 of 5 seeds.

Adding a memory path (an LSTM cell feeding the CfC, called mixed memory) raised the CfC to 0.967 ± 0.008 AUC and 0.889 ± 0.015 matched AUC, a gain of 0.037 matched AUC in 5 of 5 seeds. A plain LSTM of the same size scores 0.968 ± 0.011 and 0.879 ± 0.023, level with it, so the gain cannot yet be credited to the CfC part. The GRU still leads every CfC variant in all 5 seeds.

**Rate-shift test (November 23):** with validation packets dropped at random, the mixed-memory CfC loses less AUC than a normally trained GRU (0.835 against 0.747 at 90% dropped, 5 of 5 seeds), mostly on windows that event count already separates. A GRU trained with packets dropped matches it (AUC 0.922 against the drop-trained CfC's 0.929 at 90%, matched AUC 0.774 against 0.751). The claim we fixed in advance, that the CfC degrades less even against a drop-trained GRU, failed. Details in FINDINGS.

**Learning-rate sweep (November 24):** every model was retrained at 0.0003, 0.003 and 0.01 beside the original 0.001, keeping the rate with the best validation matched AUC. The GRU reached 0.990 AUC and 0.971 matched AUC at 0.01; the best CfC trained on complete windows reached 0.889 (CfC + memory at 0.001). Both CfCs trained on complete windows diverged one rate above their best. The GRU now leads every CfC variant in all 5 seeds. Because validation picked the rates, the challenging split is the only clean test left.

The challenging split has not been scored for any of these models.

## What is trained

| Set | Where | Arms | Seeds |
|---|---|---|---|
| Event packets | `runs/packets-2026-11-22/<arm>/` | `time_cfc`, `time_cfc_gate`, `time_cfc_const`, `time_mmcfc`, `time_mmcfc_const`, `time_mmcfc_aug`, `time_lstm`, `time_gru`, `time_gru_aug`, `time_gru_const`, `space_cfc`, `space_gru` | 0 to 4 |
| Learning-rate sweep | `runs/lr-sweep-2026-11-24/lr<rate>/<arm>/` (and `cnn/`), with `summary.json` | the six real-gap packet arms and the CNN, at 0.0003, 0.003 and 0.01 | 0 to 4 |
| Count windows | `runs/recurrent-2026-11-22/` and `runs/recurrent-40ep-2026-11-22/` (not in git) | `cfc_dt`, `cfc_const`, `gru_dt`, `gru_const` | 0 to 4 |
| Crop CNN | `runs/cnn-fred-2026-11-12/`, `runs/cnn-40ep-2026-11-22/` (not in git) | `EventCountCNN` | 0 to 4 |
| Full-frame detector | `runs/fred-detector-2026-11-12/` (not in git) | `EventDetector` | 0 to 4 |

Only the packet weights and their `results.json` files are force-added, because `.gitignore` excludes `runs/` and `*.pt`. To get the others, ask Max or rerun the training commands below.

The packet models need a window's `[gap, features, mask, log_count]` tensors, which `src/train/packets.py` builds. To load one:

```python
import torch
from src.models.packet_models import PACKET_ARMS
from src.data.event_packets import FEATURES

feature_set, constant_gap, build = PACKET_ARMS["time_cfc"]
model = build(len(FEATURES[feature_set]))
model.load_state_dict(torch.load("runs/packets-2026-11-22/time_cfc/seed_0.pt", map_location="cpu"))
```

## How to rerun

Needs the FRED recordings listed in `experiments/phase1_sequences.json` (tune and heldout) as `events.npy`, plus numpy and torch. No GPU is needed: a packet arm takes about 8 s per epoch for a CfC on Max's Mac CPU.

```
python -m src.train.recurrent --arm cfc_dt --out runs/recurrent --epochs 40
python -m src.train.packets --arm time_cfc --out runs/packets
python -m src.train.packets --arm time_gru --out runs/packets --drop-augment 0.9
python -m src.train.rate_shift --runs runs/packets-2026-11-22 --out FILE
python -m src.train.evaluate_all --set heldout --recurrent runs/recurrent-40ep-2026-11-22 --cnn runs/cnn-40ep-2026-11-22 --out FILE
```

`evaluate_all` scores the count-window models and refuses to overwrite its output file. Do not point it at `--set challenging` until the models stop changing; it is a one-time measurement.

## Next steps, in the order I would do them

1. **Done: a memory path for the CfC** (`time_mmcfc`, with the `time_lstm` control). See above and FINDINGS.
2. **Done: packet-drop test** (`src/train/rate_shift.py`, `--drop-augment` in `src/train/packets.py`). The CfC's advantage disappears against a drop-trained GRU. A remaining version to try is a real rate change, such as your own SilkyEvCam recordings or windows from far-away drones, instead of random drops.
3. **Done: learning-rate sweep** (`--lr` in `src/train/packets.py` and `src/train/cnn.py`). Next for stability: a warm-up schedule or tighter gradient clipping for every model, to see whether the CfC can train at the rates that helped the GRU.
4. **Check the cell against `ncps`** at batch size 1. It follows the published CfC equation and is tested for batching and gradients, but has not been compared with the reference code.
5. **Fix the placement shortcut before using position features.** A linear model on position and spread alone reaches 0.861 matched AUC, so those arms cannot show rotor timing is at work.
6. **Add recordings.** There are five training recordings, one per drone type. Over half the matched pairs (356 of 663) come from train/95.
7. **Then score the challenging split once** and run YOLO if the full-frame comparison is wanted (`ultralytics` is not installed and needs approval).

## Known gaps

- The CfC's gate and the GRU's feature see the gap as `log1p(ms)`; in the earlier count-window run the CfC had raw gaps. Do not compare CfC numbers across the two runs as if only the input changed.
- Window-level event count is a model input at the readout in the packet arms but not in the count-window arms.
- The camera (SilkyEvCam VGA) arrived November 18 and has no lens; nothing here has been tested on it, or on a Raspberry Pi.
- No model has been tested on birds.
