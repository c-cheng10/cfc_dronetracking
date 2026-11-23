# cfc_dronetracking

Detecting small drones with an event camera by the flicker of their spinning rotors, with a small liquid neural network (CfC) as the classifier. Built by Chris Cheng and Max toward the **IDEaS Counter-UAS (Urban) Sandbox 2027**, run by Canada's Department of National Defence.

**Core question:** can a small CfC tell a drone from background in an event stream better than simple baselines, and run fast enough on a Raspberry Pi 5?

## Where it stands (2026-11-22)

- **Phase 1 is done.** An FFT rotor-frequency detector, a drone finder that needs no labelled boxes, and a flicker filter were evaluated on FRED, including 58 recordings from its challenging split.
- **The rotor flicker is real, but it is not enough on its own.** On busy backgrounds, counting events in a box beats the FFT. Over the 58 challenging recordings, AUC was 0.789 for the event count and 0.615 for the FFT.
- **The bar for the CfC:** beat the combined FFT + count model (AUC 0.952 on the held-out set) and the event count alone (0.797 on the 9-recording challenging set, 0.789 over all 58), on the same windows.
- **Not started:** CfC training, live camera capture, and the Raspberry Pi.

Every result, including the ones against us, is in [docs/FINDINGS.md](docs/FINDINGS.md), newest first.

## How it works

1. The event camera reports each pixel's brightness change as `(x, y, polarity, t)`, with `t` in microseconds.
2. **Find** a candidate: a dense cluster of events in a 0.5 s window ([`src/phase1_find.py`](src/phase1_find.py)).
3. **Count** the events inside the box in 0.5 ms bins.
4. **Score** it: a spinning rotor makes the count pulse at the blade-pass frequency, which the FFT shows as a sharp peak ([`src/phase1_fft.py`](src/phase1_fft.py)). Peaks that stay exactly fixed for seconds, like flickering lights on mains power, are rejected.
5. **Next:** replace step 4 with a CfC trained on the same windows.

## Quick start

Needs Python 3.12. On Windows:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**1. Download one FRED recording** (about 300 MB):

```
python -c "from huggingface_hub import hf_hub_download as d; print(d('Nyquispei/FRED', 'train/36.zip', repo_type='dataset', local_dir='data/FRED'))"
python -m zipfile -e data/FRED/train/36.zip data/FRED/train/36
```

`data/` is git-ignored. If your C: drive is short on space, point it at another drive with a junction (`mklink /J data D:\cfc_data`).

**2. Decode the events.** FRED compresses its events with Prophesee's ECF codec, which plain h5py can't read. Build the decoder once (steps in [third_party/ecf/README.md](third_party/ecf/README.md)), then convert each recording:

```
python -m src.data.ecf_convert data/FRED/train/36/36/Event/events.hdf5
```

This writes `events.npy` next to the input. Open it with `np.load(path, mmap_mode="r")`. The committed build steps are for Windows only.

**3. Run the detector:**

```
python src/phase1_fft.py --seq train/36
```

Results and plots go to `experiments/`.

## Layout

```
src/data/ecf_convert.py     FRED events.hdf5 -> events.npy
src/phase1_fft.py           FFT rotor detector on labelled boxes; writes per-window CSVs
src/phase1_multiseq.py      frozen-threshold evaluation over the tune / heldout / challenging sets
src/phase1_find.py          finds drones without labels, then FFT-scores each candidate
src/phase1_example_window.py   plot of one typical drone and background window
src/models/, src/train/     CfC and GRU (empty so far)
third_party/ecf/            Prophesee's ECF decoder (Apache 2.0) plus our C wrapper
experiments/                results, frozen thresholds, per-window CSVs in experiments/csv/
docs/FINDINGS.md            the findings log
docs/HANDOFF_phase1.md      Phase 1 handoff: method and answers to review questions
notebooks/                  exploration (CfC sine test)
```

## Evaluation rules

- **Sets** are fixed in [experiments/phase1_sequences.json](experiments/phase1_sequences.json). Tune on `tune` only, then report `heldout` and `challenging` with everything frozen. `train/36` was used for development, so it is reported separately.
- **Same windows for every model** (0.5 s, 0.25 s step), so scores compare row by row. The per-window CSVs carry every baseline's score.
- **Hard negatives:** "active" backgrounds are busy image regions (trees, buildings, lights). "Shifted" backgrounds are mostly empty sky and flatter every method.
- **ML models:** at least 5 seeds, mean ± standard deviation, matched parameter counts.
- **Bin width ≤ 1 / (4 × blade-pass frequency).** FRED's DJI Mini 2 peaks near 312 Hz and the Mini 3 near 230 Hz, so 0.5 ms bins are fine for both.

## Plan

| Phase | When | Goal | Status |
|---|---|---|---|
| 0 | Nov 2–9 | Eligibility, order a camera, measure the Mini 4K's blade-pass frequency, CfC sine test | Sine test done; camera quotes sent Nov 6. Update the rest here. |
| 1 | Nov 2–20 | FFT detector on FRED, tested on held-out and challenging recordings | **Done** (Nov 9) |
| 2 | Nov 15–Dec 1 | Live camera and detector on the Mini 4K, 3-minute real video, TRL 4 application | Go/no-go due Nov 22 |
| 3 | Nov–Jan | CfC and GRU on Phase 1's windows, against the combined baseline | Started |
| 4 | Jan–Feb | Our own CfC cell, quantized, latency on a Raspberry Pi 5 | |
| 5 | Feb–Mar 10 | Field tests with the Mini 4K; reach TRL 5 | |
| 6 | Mar–Jul | Sandbox prep (Ottawa, July 26–30, 2027) | |

**IDEaS application:** due **Dec 1, 2026, 2:00 PM EDT** through the Defence Innovation Portal. It needs TRL 4 plus a plan to reach TRL 5 by Mar 10, 2027, and real footage (≤ 3 min, 1080p, MP4/MOV). Prizes go only to a Canadian incorporated entity that owns the IP, in place by July 26, 2027.

## Known limitations

- The finder keeps one candidate per window, so it can't report two drones at once.
- Ducted and very small drones (DarwinFPV, Betafpv) show weak or no rotor peak.
- Hot-pixel removal uses the whole recording, so a live system needs a calibration clip.
- Range is untested: downsampling the coordinates changed nothing, so simulating a smaller sensor needs a better model.
- Nothing has run on a live camera or a Raspberry Pi yet.

## Data

- **FRED**: Florence RGB-Event Drone dataset ([paper](https://arxiv.org/abs/2506.05163), [code](https://github.com/miccunifi/FRED)). Over 7 hours, five drone types, Prophesee IMX636 at 1280×720. Each recording has `Event/events.hdf5` plus `coordinates.txt` (`time_s: x1, y1, x2, y2, id, class`). The bundled PNG event frames are about 30 fps, too slow for rotor flicker.
- **SkyEV** (ETH, 2026): planned as an extra held-out test. Availability not yet checked.
