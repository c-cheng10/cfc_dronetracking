# CfC against GRU on event-camera drone windows

Written November 22, 2026, on the `max` branch. Training of the event-packet models described below was still running when this was committed; results will follow in `docs/HANDOFF_cfc.md`.

## The question

A CfC is a small recurrent network whose state update uses the time since its last input. We are testing whether that helps it tell a drone from busy background in an event-camera stream, compared with a GRU of the same size that is given the same time gaps. The FFT + count detector and a small CNN are the baselines.

A CfC is claimed to suit irregularly timed inputs. That does not mean it must beat a GRU on event data: a GRU can be handed the gap as an input. The comparison only says something if both models get identical inputs.

## The task

Each example is a 0.5 s window of events inside a box, from [FRED](https://github.com/miccunifi/FRED) recordings. Positives are the labelled drone boxes. Negatives are same-size "active" boxes placed over the busiest parts of the scene. Training uses five recordings, one per drone type, with 7,435 windows. Validation uses five other recordings with 8,278 windows. Recordings never appear in both. The challenging split is closed until the models stop changing, and train/36 stays development data.

## What has been tried

**1. Count windows (done).** The events in each box were counted in 0.5 ms bins and the non-empty bins were fed to the models with the gap between them. Numbers and caveats are in the November 22 entry of [FINDINGS.md](FINDINGS.md). At 40 epochs, on validation, the CfC with the real gap scored 0.953 ± 0.008 AUC against 0.967 ± 0.003 for the CNN and 0.971 ± 0.014 for the GRU.

Counting in bins removed most of the irregular timing. In validation drone windows, 96% of the gaps between non-empty bins are exactly one bin, and 48% of drone windows have no other gap. The CfC had almost nothing irregular to use. The input was also unequal: the CfC got raw gaps and the GRU got log gaps.

**2. Event packets (training now).** Each window's events are grouped into packets of N consecutive events, with N chosen so a window has at most 500 packets. A packet carries:

- the gap, in milliseconds, since the previous packet's last event
- the ON fraction (share of brightness-increase events)
- mean position and spread inside the box (in the "space" arms only)

Packet gaps are irregular by construction. Every window's packets contain exactly the number of events Phase 1 stored for it, and `src/data/event_packets.py` raises an error if they don't.

The position and spread columns are kept out of the main result on purpose. A drone box follows the annotated drone while a background box sits over scenery, so those columns could reveal the label without any rotor signal. The "time" arms, which see only the gap and the ON fraction, are the result to read. Event count is given to every model, so a model cannot win by counting alone.

## The arms

Every arm has 4,826 to 4,912 parameters and 5 seeds. All use AdamW at 0.001, batch 128, gradient norm clipped at 1, 40 epochs and the last epoch. These settings were fixed before the packet results were seen.

| Arm | Gap | What it tests |
|---|---|---|
| `time_cfc` | real, in the gate and as a feature | the main CfC |
| `time_cfc_gate` | real, in the gate only | whether the CfC needs the gap as a feature too |
| `time_cfc_const` | fixed at 1 ms | the CfC with no timing |
| `time_gru` | real, as a feature | the GRU with the same information |
| `time_gru_const` | fixed at 1 ms | the GRU with no timing |
| `space_cfc`, `space_gru` | real, plus position and spread | secondary, see above |

The CfC cell has no path that copies its state forward, which a GRU has. It has not been given one yet, so the packet run changes only the input.

## How results are read

- **AUC:** probability that a random drone window scores above a random background window. 0.5 is a coin flip.
- **Matched AUC:** the same number on 663 drone windows and 663 background windows whose event counts are nearly equal. Event count alone scores 0.502 there. 356 of the 663 pairs come from train/95 (Betafpv air75).
- **Real against constant gap:** the cost of removing timing, which is the comparison that answers the question.

## Files

| Path | What it is |
|---|---|
| `src/data/event_tokens.py`, `src/models/recurrent.py`, `src/train/recurrent.py` | step 1, the count-window models |
| `src/data/event_packets.py`, `src/models/packet_models.py`, `src/train/packets.py` | step 2, the packet models |
| `src/count_matched.py`, `src/train/evaluate_all.py` | count-matched pairs and the shared scorer |
| `src/models/cnn.py`, `src/train/cnn.py`, `src/data/cnn_windows.py` | the crop CNN baseline |
| `src/models/event_detector.py`, `src/train/event_detector.py`, `src/data/fred_detection.py`, `src/train/fred_yolo.py` | the full-frame detector (trained) and the YOLO comparison (dataset prepared, not trained) |

`data/` and `runs/` are git-ignored. Rerun step 1 with `python -m src.train.recurrent --arm cfc_dt --out runs/recurrent` and step 2 with `python -m src.train.packets --arm time_cfc --out runs/packets`.
