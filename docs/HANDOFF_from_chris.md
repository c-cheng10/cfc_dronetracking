# Handoff from Chris's lane, November 25, 2026

For Max (or Max's agent). Written on branch `chris`, which is `max` at `42a6161` plus the two commits described here. Read the **Reproduction failed** section first: it needs a decision from both of us before the CfC sweep counts.

## Summary

1. **Data matches.** `python -m src.data.fetch_fred` on Windows: 47 of 47 recordings OK, 0 `MISMATCH`, 0 `FAILED`. The 6 recordings I had from `Nyquispei/FRED` hash identically to yours from `GabrieleMagrini/FRED`.
2. **The CfC reproduction check failed, and the cause looks like CfC training instability, not setup.** Details below. Proposed test: both of us run `time_gru`.
3. **`src/train/packets.py` is about 10× faster for CfC arms** (`--parallel`, `--script`). Defaults unchanged; results identical with `--parallel`, equal to rounding with `--script`.
4. **New tests on the old split** (already in FINDINGS, November 24): the CfC cell matches `ncps` exactly, and thinning or jittering events gives the CfC no robustness advantage.
5. **Phase 3 CfC sweep is set up but paused.** Nothing from it is saved yet.

## Reproduction failed

Command from `HANDOFF_chris.md`, step 2: `time_mmcfc --split phase1 --lr 0.001`. Run with `--threads 1 --parallel 5 --script` (Windows, torch 2.15.0+cpu, 22 logical cores).

| | Matched AUC | AUC | Matched AUC by seed 0–4 |
|---|---|---|---|
| Chris | 0.848 ± 0.037 | 0.964 ± 0.008 | 0.877, 0.883, 0.795, 0.861, 0.826 |
| Max | 0.889 ± 0.015 | 0.967 ± 0.008 | 0.862, 0.897, 0.892, 0.898, 0.895 |

That's 0.041 below, about 2.7 of your standard deviations, so the handoff says stop and compare. What I found:

- **Plain AUC and the parameter count (4,823) match.** Three seeds land inside your range. Two seeds make the gap.
- **The loss curves show instability.** Seed 2 never converged (final loss 0.594, against about 0.39 for good seeds). Seed 4 spiked to 0.85 at epoch 25 and finished at 0.497. Your seeds spiked to 0.75–0.80 too, but all recovered.
- **The same seed moves with tiny numeric changes.** Seed 0 gave 0.851 (2 threads, original code), 0.890 (8 threads) and 0.877 (1 thread + TorchScript). Thread count alone changes float summation order.

**Reading:** the CfC's training is chaotic enough that rounding differences between machines decide whether a seed recovers from a spike. A seed-mean ± std from one machine is therefore not a stable property of the CfC.

**Proposed next steps, needing agreement:**
1. **Both run `time_gru --split phase1 --lr 0.001`.** The GRU is stable, so a match confirms the shared setup and isolates the gap to the CfC. Chris's side takes about 20 minutes.
2. **Fix training stability for every model before the phase 3 CfC sweep counts.** This is item 1 of "What to try first": warm-up, tighter clipping, or a decay schedule. Otherwise the sweep's learning-rate choice partly measures seed luck.
3. **Consider reporting more seeds (10) for CfC arms,** or the fraction of seeds that converge, since the spread is the finding.

## Changes to `src/train/packets.py` (shared script, please review)

- **`--parallel N`:** trains seeds in N processes. Each process loads the data once (`setup`, cached), and `train_seed` holds the old per-seed loop body unchanged. The 1-epoch check against the old code at `--threads 1` gave **identical** losses and AUCs for all 5 seeds.
- **`--script`:** `torch.jit.script(model)`. On the same check the differences were 2.3e-6 matched AUC and 6.9e-7 loss. Saved weights load into the normal eager model.
- **`predict` uses `torch.no_grad()` instead of `torch.inference_mode()`.** TorchScript reuses a graph profiled under inference mode, and the next seed's training then fails ("Inference tensors cannot be saved for backward"). Results are unaffected. `rate_shift.py` and `event_shift.py` call `predict` and were not rerun after this change.
- **`results.json` also records `threads` and `script`.**
- **Defaults keep the old behaviour** (serial, eager, 2 threads).

Benchmark, `time_mmcfc`, one batch of 128 × 500 packets forward + backward, 5 processes at once, seconds per batch:

| Threads per process | Eager | Scripted (+ the 4 output layers merged) |
|---|---|---|
| 1 | 0.45 | 0.35 |
| 2 | 0.52 | 0.43 |
| 4 | 0.77 | 0.56 |

**What this shows:** the CfC is limited by per-step Python and dispatch overhead (about 40,000 sequential cell steps per epoch), so extra threads hurt. Five seeds at 1 thread each is the best layout. The reproduction check took 20 minutes this way, against an estimated 4+ hours serially at 2 threads. The merged-layers variant was benchmarked only; it is **not** in the code, since it would change the CfC model file. `torch.compile` is unavailable on this machine (no MSVC). `--script` has only been tested on `time_mmcfc`, not on the GRU or LSTM arms or with `--drop-augment`.

## Results already in FINDINGS (November 24)

- **`python -m src.models.check_cfc_ncps`:** `CfCCell` matches `ncps` to 0.0 over 200 steps at batch size 1, with random weights and with trained `time_cfc` seed 0. Mapping: a → −a, f1 ↔ f2, b unchanged.
- **`python -m src.train.event_shift --runs runs/packets-2026-11-22 --out FILE`:** thins events (keep 50%, 25%, 10%) or jitters timestamps (0.1, 0.5, 2 ms), then re-packetizes. Old split, lr 0.001 models. The pre-registered claim (the CfC degrades less) **failed**:

| Arm | None | Thin to 10% | Jitter 2 ms |
|---|---|---|---|
| CfC | 0.960 / 0.851 | 0.918 / 0.668 | 0.955 / 0.835 |
| mmCfC | 0.967 / 0.889 | 0.900 / 0.698 | 0.964 / 0.874 |
| LSTM | 0.968 / 0.879 | 0.911 / 0.675 | 0.965 / 0.865 |
| GRU | 0.979 / 0.911 | 0.860 / 0.721 | 0.976 / 0.895 |
| GRU, drop-trained | 0.967 / 0.897 | 0.891 / 0.755 | 0.965 / 0.880 |
| mmCfC, drop-trained | 0.965 / 0.884 | 0.897 / 0.756 | 0.964 / 0.871 |

Cells are AUC / matched AUC.

- **Packetization absorbs real rate drops.** Thinning to 25% moved train/36's median packet gap only from 0.95 to 1.02 ms, so random packet drops (`packet_drop.py`) are a harsher, less realistic shift than event thinning.
- **The plain GRU is the fragile one on plain AUC** (−0.12 at 10%), but it leads on matched AUC at every thinning level.
- **Not yet done:** rerun `event_shift.py` on phase 3 with your lr 0.01 GRU, and add a `--split` flag to it like `rate_shift.py`.

## Phase 3 CfC sweep (handoff item 2): paused

- **Plan:** `time_mmcfc` and `time_mmcfc --drop-augment 0.9` at 0.0003, 0.001, 0.003 and 0.01, `--split phase3 --threads 1 --parallel 5 --script`. Output goes to `runs/phase3-cfc-2026-11-25/lr<rate>/`, then `rate_shift.py --split phase3` on each rate folder.
- **Status:** the phase 3 packet and window caches are built (690 MB in `runs/window-cache`, prebuilt one sequence per process so parallel workers never race on a cache file). 3 of 8 runs started and are paused mid-training. Nothing is saved yet.
- **Will be resumed or restarted** once the stability question above is settled.
- **Cache staleness:** `packets_*.npz` still has no staleness key (unlike `window_cache`). If `event_packets.py` changes, delete the packet caches by hand.

## Files

| File | State |
|---|---|
| `src/models/check_cfc_ncps.py`, `src/train/event_shift.py`, `src/data/event_packets.py` (`build_sequence(shifts=...)`, default unchanged), FINDINGS Nov 24 entry | commit `94d2c08` |
| `src/train/packets.py` (`--parallel`, `--script`, `no_grad`), this file | next commit on `chris` |

The pull request from `chris` into `max` is not opened yet. Open it at https://github.com/c-cheng10/cfc_dronetracking/compare/max...chris.
