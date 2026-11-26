# Handoff from Chris's lane, November 25, 2026

For Max (or Max's agent). Branch `chris` is `max` at `6af16fb` plus the commits listed under **Files**. Read **Questions for Max** first; several items below wait on an answer.

## Summary

1. **Data matches.** `python -m src.data.fetch_fred` on Windows: 47 of 47 recordings OK, 0 `MISMATCH`, 0 `FAILED`. The 6 recordings Chris already had from `Nyquispei/FRED` hash identically to yours from `GabrieleMagrini/FRED`.
2. **The CfC reproduction check failed, but the GRU check reproduces** (0.909 ± 0.010 matched AUC against your 0.911 ± 0.006). The shared setup is fine, and the CfC gap is its unstable training.
3. **`src/train/packets.py` is about 10× faster for CfC arms** (`--parallel`, `--script`), **checkpoints every epoch** (`--resume`), and the packet and window caches are now written atomically. Defaults are unchanged.
4. **Earlier results** (in FINDINGS, November 24): the CfC cell matches `ncps` exactly, and thinning or jittering events gives the CfC no robustness advantage.
5. **The phase 3 CfC sweep is ready to run but not started.** A first attempt was killed; nothing from it is kept.
6. **Tracking groundwork** (FINDINGS, November 25, Chris):
   - The GRU holds up on the finder's loose boxes: it detects 66% of drone-windows with 20 false alarms per frame-minute, against the FFT's 58% and 66.
   - The finder is the bottleneck: it boxes only 73% of drones, and 53% of DJI Mini 2.
   - Alarm voting cuts alarm episodes 4 to 6 fold but not alarm time.
   - False alarms are concentrated in two recordings, train/18 and train/13.

## Questions for Max

1. **`EventDetector` (`runs/fred-detector-2026-11-12/`):** what AP50 and AP75 did it reach? Nothing in FINDINGS, the README or the commit records a score, and the weights exist only on your machine.
2. **Your commit `1c50854` says "Trained on validation recordings only."** Does that mean it trained on the recordings we validate on? If so, its validation score can't be used, and it would need retraining on the training recordings before it can feed a tracker.
3. **Reproduction: answered from Chris's side.** `time_gru` gave 0.909 ± 0.010 matched AUC against your 0.911 ± 0.006, so no action is needed unless you want to rerun `time_mmcfc` with `--threads 1 --parallel 5` to see your spread under those settings.
4. **Stability fix:** which change do we try for every model: linear warm-up, a clip of 0.5 instead of 1, cosine decay, or a combination? The phase 3 CfC sweep waits on this, or runs as is and reports the instability. Which do you prefer?
5. **Shared-script changes:** OK to merge the `packets.py`, `event_packets.py` and `window_cache.py` changes below into `max`? They change no result at default settings.
6. **Model bundle for tracking:** can each trained packet model save the training log-count mean and std and its alarm threshold beside its weights? The readout needs the count standardised with *training* statistics, and today they are recomputed from the training recordings. A tracker shouldn't need the training data to load a model.
7. **Track-level metrics:** do you agree to fix these before building anything? Share of drones that get a confirmed track, seconds to first confirmation, false confirmed tracks per minute, and ID switches.

## Reproduction failed

Command from `HANDOFF_chris.md` step 2 (`time_mmcfc --split phase1 --lr 0.001`), run with `--threads 1 --parallel 5 --script`. Windows, torch 2.15.0+cpu, 22 logical cores.

| | Matched AUC | AUC | Matched AUC by seed 0–4 |
|---|---|---|---|
| Chris | 0.848 ± 0.037 | 0.964 ± 0.008 | 0.877, 0.883, 0.795, 0.861, 0.826 |
| Max | 0.889 ± 0.015 | 0.967 ± 0.008 | 0.862, 0.897, 0.892, 0.898, 0.895 |

That's 0.041 below, about 2.7 of your standard deviations, so the handoff says stop and compare. What we found:

- **Plain AUC and the parameter count (4,823) match.** Three seeds land inside your range, and two seeds make the whole gap.
- **The loss curves show instability.** Seed 2 never converged: its final loss was 0.594, against about 0.39 for good seeds. Seed 4 spiked to 0.85 at epoch 25 and finished at 0.497. Your seeds spiked to 0.75–0.80 too, but all recovered.
- **The same seed moves with tiny numeric changes.** Seed 0 gave 0.851 with 2 threads and the original code, 0.890 with 8 threads, and 0.877 with 1 thread plus TorchScript. The thread count alone changes the order floats are summed in.

**Reading:** the CfC trains chaotically enough that rounding differences decide whether a seed recovers from a loss spike. A seed mean ± std from one machine is therefore not a stable property of the CfC.

**Confirmed with the GRU** (`runs/check-gru`, `--threads 1 --parallel 5`): 0.909 ± 0.010 matched AUC and 0.9791 AUC, against your 0.911 ± 0.006 and 0.9793. The shared setup reproduces.

## Changes to shared code (question 5)

### `src/train/packets.py`
- **`--parallel N`:** trains the seeds in N processes. Each process loads data once (`setup`, cached), and `train_seed` holds the old per-seed loop. Checked: identical losses and AUCs to the old code at `--threads 1`.
- **`--script`:** `torch.jit.script(model)`. It matches eager to about 2e-6 matched AUC, and its saved weights load into the normal eager model. Only tested on `time_mmcfc`.
- **Checkpoints:** every epoch, each seed writes `checkpoint_seed_N.pt` (model, optimizer, torch and augmentation RNG state) through a temp file and a rename. `--resume` continues an interrupted run in the same `--out`, and it refuses a checkpoint made with different settings.
  - Checked: a 1-epoch run resumed to 2 epochs equals an uninterrupted 2-epoch run **exactly** on all 5 seeds with `--drop-augment 0.9`.
  - With `--script` the match is only to about 1e-9, because TorchScript re-optimises its graph in each new process.
- **`predict` uses `torch.no_grad()` instead of `torch.inference_mode()`.** TorchScript reuses a graph profiled under inference mode, and the next seed's training then fails with "Inference tensors cannot be saved for backward". This does not change results. `rate_shift.py`, `event_shift.py` and `sparsity_slice.py` were not rerun after it.
- **`results.json` also records `threads` and `script`.**

### `src/data/event_packets.py` and `src/data/window_cache.py`
- **Caches are written to a per-process temp file, then renamed.** Before this, parallel workers building the same missing cache could read each other's half-written `.npz`. This never bit us, because Chris prebuilt the phase 3 caches one sequence per process.
- **Still open:** `packets_*.npz` has no staleness key (unlike `window_cache`). Delete them by hand if `event_packets.py` changes. The window cache is keyed on the CSV's mtime, so it rebuilds on any machine where the CSVs have a different mtime, which needs `events.npy`.

### Speed, `time_mmcfc`
One batch (128 windows × 500 packets) forward and backward, 5 processes at once, seconds per batch:

| Threads per process | Eager | Scripted, with the 4 output layers merged |
|---|---|---|
| 1 | 0.45 | 0.35 |
| 2 | 0.52 | 0.43 |
| 4 | 0.77 | 0.56 |

- **The CfC is limited by per-step Python and dispatch overhead,** about 40,000 sequential cell steps per epoch. Extra threads hurt, so 5 seeds at 1 thread each is the best layout.
- **The reproduction check took 20 minutes this way,** against an estimated 4+ hours serially at 2 threads.
- **The merged-layers variant was benchmarked only.** It is not in the code, because it would change the CfC model file.
- **`torch.compile` can't run here** (no MSVC), and a GPU doesn't help a model this small.

## Results already in FINDINGS (November 24)

- **`python -m src.models.check_cfc_ncps`:** `CfCCell` matches `ncps` to 0.0 over 200 steps at batch size 1, with random weights and with trained `time_cfc` seed 0. The mapping: a → −a, f1 ↔ f2, b unchanged.
- **`python -m src.train.event_shift --runs runs/packets-2026-11-22 --out FILE`:** thins events (keep 50%, 25%, 10%) or jitters timestamps (0.1, 0.5, 2 ms), then re-packetizes. This was run on the old split with the lr 0.001 models. The claim fixed in advance, that the CfC degrades less, **failed**. AUC / matched AUC:

| Arm | None | Thin to 10% | Jitter 2 ms |
|---|---|---|---|
| CfC | 0.960 / 0.851 | 0.918 / 0.668 | 0.955 / 0.835 |
| mmCfC | 0.967 / 0.889 | 0.900 / 0.698 | 0.964 / 0.874 |
| LSTM | 0.968 / 0.879 | 0.911 / 0.675 | 0.965 / 0.865 |
| GRU | 0.979 / 0.911 | 0.860 / 0.721 | 0.976 / 0.895 |
| GRU, drop-trained | 0.967 / 0.897 | 0.891 / 0.755 | 0.965 / 0.880 |
| mmCfC, drop-trained | 0.965 / 0.884 | 0.897 / 0.756 | 0.964 / 0.871 |

- **This agrees with your November 25 sparsity slice.** Thinning to 25% moved train/36's median packet gap only from 0.95 to 1.02 ms, and you found the GRU keeps 0.938 AUC on the sparsest real windows. Random packet drops are a harsher, less realistic shift than real event loss. Event thinning is probably the better robustness test and the better augmentation.
- **Not yet done:** add a `--split` flag to `event_shift.py` and rerun it on phase 3 with your lr 0.01 GRU.

## Phase 3 CfC sweep: ready, not started

- **Plan** (`HANDOFF_chris.md` item 2): `time_mmcfc` and `time_mmcfc --drop-augment 0.9` at 0.0003, 0.001, 0.003 and 0.01, with `--split phase3 --threads 1 --parallel 5 --script --resume`, 3 runs at a time. Output goes to `runs/phase3-cfc-2026-11-25/lr<rate>/`, followed by `rate_shift.py --split phase3` on each rate folder.
- **Estimated time:** 6–9 hours on Chris's laptop.
- **Status:** the phase 3 packet and window caches are built (690 MB in `runs/window-cache`). A first attempt was killed before any run finished, and nothing from it is kept.
- **Waiting on question 4.** The concern: with seeds this unstable, a learning rate can look bad because its seeds stumbled, not because the rate is wrong.

## Tracking: what exists and what doesn't

Today the GRU answers yes or no for a box it is given, and the boxes come from FRED's labels. Nothing finds, confirms and follows a drone end to end.

| Piece | State |
|---|---|
| Box finder, `find_candidates` in `src/phase1_find.py` | Works, but proposes **only one box per window** (`max_cands: 1`), so it can never see two drones |
| `EventDetector` (full-frame CNN) | Trained November 12, **no score recorded** (questions 1 and 2) |
| GRU classifier | Best model, only ever run on FRED's boxes |
| Tracker (follow a drone from window to window) | Doesn't exist |
| Track-level decision (for example 3 of the last 4 windows say drone) | Doesn't exist |
| Ground truth for scoring tracks | Exists: `coordinates.txt` gives each box a track ID, read by `load_tracks` in `phase1_find.py` |

### Box finder recall
From `experiments/phase1_find_results.txt` (Phase 1 run, old split, not rerun). A drone counts as found when IoU with its swept box is above 0.1.

| Drone type (old validation recordings) | Found |
|---|---|
| DJI Tello | 96% |
| DarwinFPV | 89% |
| DJI Mini 3 | 88% |
| DJI Mini 2 | 76% |
| Betafpv | 27% |
| All | 77% |

- **Before scoring, it proposes 53 wrong boxes per minute.** The FFT cut that to 21 per minute. The GRU has not been tried as the scorer.
- **On challenging recordings it finds 16–77% of drones.** The Tello falls from 96% to 17%.
- **Its boxes are loose:** IoU above 0.1 counts a partial hit, and each box spans the whole 0.5 s path. A GRU trained on tight FRED boxes may score them worse. That's untested.

### Recordings with two drones at once
10 of the 47. FRED gives each drone its own ID, so ID switches can be measured.

| Recording | Split | Frames with both drones |
|---|---|---|
| train/211, 213, 49 | training | 90–94% |
| train/215, 106, 32, 44 | training | 22–61% |
| train/31 | validation | 14% |
| train/35 | test (closed) | 74% |
| train/36 | development | 52% |

The tracker isn't learned, so it can be developed on the training two-drone recordings without leaking anything into the GRU's validation scores. Check it on train/36 and train/31, and keep train/35 for the one-time test.

### Measured November 25 (FINDINGS, Chris)
- **`src/track_vote.py`:** voting "3 of 4" on FRED's boxes cuts the drop-trained GRU's alarm episodes from 3.23 to 0.75 per box-minute, but false-alarm windows only from 8.96 to 5.58 (claim: at least 5-fold, **failed**).
  - False alarms come in long runs on a few boxes. 68% come from train/18 and train/13.
  - Can you look at those recordings' active background boxes, or tell us what's in that scene?
- **`src/finder_gru.py`:** with up to 3 boxes per window, the finder boxes 73% of present drone-windows.
  - The GRU detects 66% (plain) or 65% (drop-trained) of them, with about 20 false alarms per frame-minute. The FFT on the same boxes gets 58% with 66 false alarms.
  - Box AUC is 0.95 on finder boxes, against 0.979 on FRED's.
  - The finder boxes only 53% of DJI Mini 2 drone-windows.

### Proposed order (no tracker written yet)
1. **Done:** voting and finder-plus-GRU, above.
2. **The model bundle** (question 6).
3. **Offline pipeline on FRED:** finder or `EventDetector` → GRU → tracker, scored with the metrics in question 7.
4. **Live:** SilkyEvCam stream and a Raspberry Pi timing test.

**Dependencies:** `scipy` is not installed. With at most two drones, greedy nearest-box matching plus a NumPy Kalman filter is enough, so it may not be needed.

## Files

| File | Commit on `chris` |
|---|---|
| `src/models/check_cfc_ncps.py`, `src/train/event_shift.py`, `src/data/event_packets.py` (`build_sequence(shifts=...)`), FINDINGS Nov 24 entry | `94d2c08` |
| `src/train/packets.py` (`--parallel`, `--script`, `no_grad`), first version of this file | `62e9207` |
| `src/train/packets.py` (checkpoints, `--resume`), atomic cache writes in `event_packets.py` and `window_cache.py` | `85b8878` |
| merge of `max` at `6af16fb` | `27c54c1` |
| this version of the handoff | not committed yet |

The pull request from `chris` into `max` is not open yet. Open it at https://github.com/c-cheng10/cfc_dronetracking/compare/max...chris
