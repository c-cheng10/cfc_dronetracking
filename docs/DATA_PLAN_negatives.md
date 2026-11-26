# Harder negatives: data plan

Written November 25, 2026. Nothing here has been downloaded. Each source needs Max's approval before it is fetched.

## Why

Our background windows come from the same FRED recordings as the drones: trees, buildings and lights near the flight path. No bird, insect or busy street is in any training or validation set, and a bird is the first false alarm a reviewer will ask about.

These would go into a separate **stress set**, scored after training and never used to choose learning rates or models. Keeping it separate leaves the Phase 3 validation numbers comparable with what Max and Chris have already run.

## Candidates

| Source | What it adds | Format and size | Licence | Catch |
|---|---|---|---|---|
| [EV-Flying](https://arxiv.org/abs/2506.04048) (2025) | Birds, insects and drones with boxes and track IDs, 27 recordings, over an hour | Prophesee EVK2, 1280 × 720, raw events | Not checked | No public download found. Request it from the FRED lab (same group). Best fit for birds. |
| Our SilkyEvCam | Static views of streets, trees, birds near feeders, at our own resolution | 640 × 480 | Ours | We have to record and label it. The camera works on the Windows laptop; it still needs a lens. |
| [DSEC](https://dsec.ifi.uzh.ch/) (2021) | Busy urban scenes, many moving objects | VGA event cameras, per-sequence zips of about 0.8 to 2.4 GB | CC BY-SA 4.0, commercial use allowed | Mounted on a moving car, so the whole scene moves. Our node is static, so these backgrounds are harder than anything a fixed node sees. Useful as a worst case only. |
| [EV-UAV](https://github.com/ChenYichen9527/Ev-UAV) (2025) | Small, distant drones in urban clutter | 346 × 260 | Bans commercial use | Timestamps in milliseconds, too coarse for our packet gaps. Science fair only. |
| [GRIFFIN](https://arxiv.org/abs/2101.10371) (2021) | Event camera on a flapping-wing robot | Onboard sensor data | Not checked | Recorded from the ornithopter, not of birds. Not a fit. |

## How a source joins the stress set

1. Max approves the download, by name and size.
2. Convert its events to our `events.npy` format (x, y, p, t in microseconds).
3. Cut background boxes the same way Phase 1 does (`src/phase1_fft.py`), and drone or bird boxes from its labels.
4. Record its recordings and checksums in a new `experiments/stress_sequences.json`, so Chris gets the same files.
5. Score every trained model on it once each model is final, reporting false alarms per background-box minute as well as AUC.

## Recommended order

1. Request EV-Flying (birds) now, because a reply may take weeks.
2. Record our own urban and bird backgrounds once the lens arrives; that is also the camera IDEaS will see.
3. Add one or two DSEC sequences as a worst case, if a moving background is worth showing.
