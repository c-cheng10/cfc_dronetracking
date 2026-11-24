# Findings log

A running record of what we have measured, newest first. Add a dated entry for each result, including the ones that go against us, and never edit an old entry to change what it said: if a later result overturns it, add a new entry and mark the old one superseded.

Each entry says what was run, the numbers, and where the files are. AUC is the probability that a random drone window scores higher than a random background window (0.5 = coin flip, 1.0 = perfect).

---

## November 23, 2026: dropping packets: the CfC degrades less than a plain GRU, not less than an augmented one

**Claim tested** (fixed before any result): trained at normal event rates, the CfC loses less accuracy than the GRU when validation packets are dropped, even against a GRU trained with packets dropped.

**Run:** `src/train/rate_shift.py`. Validation packets were removed at random at 0, 25, 50, 75 and 90%; a surviving packet's gap becomes the time since the previous survivor, and the event count is recomputed (`src/data/packet_drop.py`). Every arm sees the same dropped packets. Two new arms were trained with `--drop-augment 0.9` (each window loses packets at a rate drawn from 0 to 90%, freshly each epoch): `time_gru_aug` and `time_mmcfc_aug`. Everything else matches the entries below. At 90% the median drone gap is 6.86 ms, too coarse to sample a 200 Hz rotor, so that rate tests how each model degrades, not rotor detection.

AUC, mean of 5 seeds:

| Arm | 0% | 25% | 50% | 75% | 90% |
|---|---:|---:|---:|---:|---:|
| GRU | 0.979 | 0.919 | 0.822 | 0.811 | 0.747 |
| CfC | 0.960 | 0.927 | 0.872 | 0.860 | 0.804 |
| Mixed-memory CfC | 0.967 | 0.928 | 0.855 | 0.863 | 0.835 |
| LSTM | 0.968 | 0.942 | 0.891 | 0.877 | 0.740 |
| GRU, constant gap | 0.969 | 0.956 | 0.938 | 0.915 | 0.890 |
| CfC, constant gap | 0.945 | 0.939 | 0.932 | 0.920 | 0.899 |
| Mixed-memory CfC, constant gap | 0.964 | 0.957 | 0.945 | 0.924 | 0.890 |
| GRU, drop-augmented | 0.967 | 0.964 | 0.957 | 0.945 | 0.922 |
| Mixed-memory CfC, drop-augmented | 0.965 | 0.963 | 0.959 | 0.947 | 0.929 |

Matched AUC, same order: GRU 0.911 / 0.864 / 0.861 / 0.822 / 0.537; CfC 0.851 / 0.829 / 0.799 / 0.707 / 0.547; mixed-memory CfC 0.889 / 0.865 / 0.836 / 0.788 / 0.550; LSTM 0.879 / 0.848 / 0.831 / 0.760 / 0.468; augmented GRU 0.897 / 0.889 / 0.877 / 0.851 / 0.774; augmented mixed-memory CfC 0.884 / 0.882 / 0.869 / 0.826 / 0.751. Every number per seed is in `runs/packets-2026-11-22/rate_shift.json`.

**What it means:**

- **The claim fails as stated.** Against the drop-augmented GRU, the drop-augmented CfC is level (AUC 0.929 against 0.922 at 90%, matched AUC 0.751 against 0.774, the GRU ahead in 5 of 5 seeds), and the unaugmented CfC is far below it (AUC −0.101 at 50%, 0 of 5 seeds).
- **Against an unaugmented GRU, the CfC holds AUC better.** Mixed-memory CfC loses 0.100 less AUC than the GRU at 90% (5 of 5 seeds), 0.046 less at 50% (5 of 5). On matched AUC the difference is small and mixed: 0.036 less loss at 90% (4 of 5 seeds), none at 50% or 75%. So most of the CfC's advantage is in windows that event count already separates, not in the hard pairs.
- **Real gaps make every model fragile when gaps change.** Each constant-gap arm degrades less than its real-gap version, e.g. GRU 0.890 against 0.747 at 90%. A model that learned the training gap distribution fails when it shifts; the CfC's gate softens that but does not remove it.
- **Training with dropped packets fixes it for both models,** at a cost of 0.012 AUC for the GRU at 0% (0.979 to 0.967).
- AUC for the unaugmented CfCs rises from 50% to 75% (0.855 to 0.863). The cause has not been checked.

**For the pitch:** the CfC's timing assumption is a real but small advantage that data augmentation matches. The defensible statement is that a CfC degrades more gracefully than a GRU trained the ordinary way. It is not better than a GRU trained with dropped packets.

---

## November 22, 2026: a memory path lifts the CfC to the LSTM, still below the GRU

**Run:** `src/train/packets.py` with three new time-only arms, same packets, protocol and 663 matched pairs as the entry below. `time_mmcfc` is a mixed-memory CfC: an LSTM cell carries a memory and its state enters the CfC at every packet (hidden 20, backbone 27). `time_lstm` is a plain LSTM (hidden 33), added so a gain cannot be credited to the CfC when it comes from the LSTM. Parameters: 4,823 and 4,952. The challenging split is not scored.

| Arm (validation) | AUC | Matched AUC | Recall | False alarms/min |
|---|---:|---:|---:|---:|
| GRU, real gap (from below) | 0.979 ± 0.001 | 0.911 ± 0.006 | 0.910 | 10.2 |
| Mixed-memory CfC, real gap | 0.967 ± 0.008 | 0.889 ± 0.015 | 0.826 | 7.9 |
| LSTM, real gap | 0.968 ± 0.011 | 0.879 ± 0.023 | 0.860 | 9.4 |
| CfC, real gap (from below) | 0.960 ± 0.011 | 0.851 ± 0.006 | 0.825 | 8.8 |
| Mixed-memory CfC, constant gap | 0.964 ± 0.011 | 0.834 ± 0.032 | 0.844 | 8.6 |

Differences, matched AUC, paired by seed: mixed-memory CfC over plain CfC +0.037 (5 of 5 seeds); over its constant-gap version +0.054 (5 of 5); over the LSTM +0.010 (2 of 5); against the GRU −0.022 (0 of 5).

**What it means:**

- **The memory path helps the CfC,** by 0.037 matched AUC in every seed, and timing still matters inside it.
- **It does not separate the CfC from an LSTM.** The mixed-memory CfC and the LSTM are level (0.889 against 0.879, 2 of 5 seeds in the CfC's favour), so this run cannot credit the gain to the CfC part.
- **The GRU still leads** every CfC variant, by 0.022 matched AUC and 0.012 AUC in all 5 seeds, and has the lowest seed spread.
- **Training is steadier but not steady.** The mixed-memory CfC's loss peaked at 0.80 after epoch 5 (plain CfC: 1.39). Its constant-gap version peaked at 1.57. The learning rate has still not been varied for any arm.
- Cost: 12.5 s per epoch on the Mac CPU, against about 8 s for the plain CfC.

**Files:** `runs/packets-2026-11-22/time_mmcfc*/` and `time_lstm/` (not committed).

---

## November 22, 2026: event packets help the CfC, but the GRU still leads

**Run:** `src/train/packets.py`, seven arms, 40 epochs, 5 seeds, settings fixed before any result. Packets hold N consecutive events and their real gap; only the arms marked "space" also see position and spread. Every window is verified against its stored event count. Same validation windows and 663 matched pairs as the entry below. The challenging split is not scored.

| Arm (validation) | AUC | Matched AUC | Recall | False alarms/min |
|---|---:|---:|---:|---:|
| GRU, real gap | 0.979 ± 0.001 | 0.911 ± 0.006 | 0.910 | 10.2 |
| GRU, constant gap | 0.969 ± 0.018 | 0.815 ± 0.123 | 0.874 | 11.3 |
| CfC, real gap (gate and feature) | 0.960 ± 0.011 | 0.851 ± 0.006 | 0.825 | 8.8 |
| CfC, real gap (gate only) | 0.957 ± 0.020 | 0.815 ± 0.033 | 0.830 | 8.4 |
| CfC, constant gap | 0.945 ± 0.002 | 0.786 ± 0.061 | 0.809 | 8.9 |
| GRU, position and spread added | 0.996 ± 0.003 | 0.992 ± 0.001 | 0.985 | 7.7 |
| CfC, position and spread added | 0.996 ± 0.001 | 0.989 ± 0.003 | 0.991 | 11.2 |

For comparison on the same validation windows: CNN 0.967 ± 0.003 / 0.853 ± 0.018, FFT + count 0.952 / 0.608, count-window CfC 0.953 ± 0.008 / 0.756 ± 0.009.

**What it means:**

- **The GRU beats the CfC on identical packets:** by 0.020 AUC (all 5 seeds) and 0.060 matched AUC (all 5 seeds). The CfC's 0.851 matched AUC is level with the CNN's 0.853, not above it.
- **Timing helps the CfC.** Real gaps beat constant gaps by 0.015 AUC (4 of 5 seeds) and 0.066 matched AUC (5 of 5). Giving the gap only to the gate is 0.036 worse than also giving it as a feature (4 of 5 seeds).
- **Timing helps the GRU too:** 0.096 matched AUC (5 of 5 seeds) with a large spread, because one constant-gap seed reached only 0.596.
- **The CfC trains less steadily.** Its loss spiked up to 1.39 after epoch 5, against 0.75 for the GRU, and the last epoch is kept without selection. A lower learning rate or a decay schedule has not been tried; it would have to go to every arm.
- **Read the space arms as untrusted.** A linear model on the average position and spread, with no timing, scores 0.918 AUC and 0.861 matched AUC on validation. Drone boxes sit on the annotated drone and background boxes sit on scenery, so part of that signal comes from how the boxes were placed. Nothing here separates that from real shape.

**Not done:** the challenging split, YOLO, an `ncps` check, and any change to the CfC cell.

**Files:** `runs/packets-2026-11-22/` (weights and results per arm, force-added to the `max` branch; logs and raw scores are not).

---

## November 22, 2026: first CfC loses to the GRU and the CNN

**Run:** `src/train/recurrent.py` trained a CfC and a GRU, each with the real gap between non-empty 0.5 ms bins and with the gap fixed at 1. Windows, training recordings and validation recordings are the CNN's. Every model has 4,833 to 4,879 parameters and 5 seeds. The recurrent models use AdamW at 0.001, batch 128, gradient norm clipped at 1 and the last epoch. The 12-epoch run was fixed before any result was seen. The 40-epoch run was added after validation showed training loss still falling, so it is secondary; the CNN was retrained at 40 epochs too.

"Matched AUC" compares 663 drone windows with 663 background windows from the same recordings, paired so log event counts differ by under 0.15 (`src/count_matched.py`). Event count alone scores 0.502 on it. 356 of the 663 pairs come from train/95 (Betafpv air75), so this number leans on one recording.

| Model (validation) | AUC, 12 epochs | Matched, 12 | AUC, 40 epochs | Matched, 40 |
|---|---:|---:|---:|---:|
| Event count | 0.932 | 0.502 | | |
| FFT + count | 0.952 | 0.608 | | |
| Inverted variance / mean of bin counts | 0.284 | 0.786 | | |
| CNN | 0.957 ± 0.002 | 0.735 ± 0.019 | 0.967 ± 0.003 | 0.853 ± 0.018 |
| CfC, real gap | 0.942 ± 0.005 | 0.680 ± 0.015 | 0.953 ± 0.008 | 0.756 ± 0.009 |
| CfC, constant gap | 0.939 ± 0.001 | 0.644 ± 0.014 | 0.944 ± 0.003 | 0.703 ± 0.015 |
| GRU, real gap | 0.954 ± 0.007 | 0.767 ± 0.039 | 0.971 ± 0.014 | 0.867 ± 0.089 |
| GRU, constant gap | 0.953 ± 0.003 | 0.765 ± 0.011 | 0.964 ± 0.001 | 0.864 ± 0.016 |

At 40 epochs GRU seed 0 with the real gap had a training loss spike (1.195 near epoch 16) and ended at 0.708 matched AUC; the other four seeds scored 0.904 to 0.909. Seed 0 is kept in the mean, which is why the standard deviation is 0.089.

**What it means:**

- **The CfC target is missed.** At 40 epochs its AUC (0.953 ± 0.008) is level with the frozen FFT + count baseline (0.952) and below the CNN and both GRUs.
- **The real gap helps the CfC,** by 0.036 matched AUC at 12 epochs and 0.053 at 40. For the GRU the difference is 0.002 and 0.003, with the 40-epoch figure inside the seed spread.
- **Shuffling the bins lowers the 40-epoch CNN's matched AUC** from 0.853 to 0.786, the same as variance/mean. At 12 epochs shuffling changed nothing (0.735 to 0.730). Shuffled windows are unlike anything in training, so this does not separate bin order from unfamiliar input.
- **The setup was not equal.** The CfC received raw gaps of 1 to 1,000 and the GRU received log gaps. The CfC cell also has no path that copies its state forward, which the GRU has. Neither explanation has been tested.

**Not done:** the challenging split has not been scored for the CNN, CfC or GRU, and stays closed until the models stop changing. YOLO11n is untrained (`ultralytics` is not installed). The CfC cell follows the published equation and has not been checked against `ncps`.

**Files:** `runs/recurrent-2026-11-22/`, `runs/recurrent-40ep-2026-11-22/` and `runs/cnn-40ep-2026-11-22/` (git-ignored); `heldout.json` in the first two has every model, seed and recording.

---

## November 9, 2026 · Plain FFT trails event count across 58 challenging recordings

Phase 1 tested whether a sharp repeating frequency inside a labelled drone box could distinguish drone windows from background windows. It also established a baseline for later models. The experiment produced precision and recall measurements on FRED, which was the Phase 1 completion criterion in the [README](../README.md#plan-timeline-set-2026-11-01).

The run evaluated the 49 remaining clips from the [FRED challenging split](https://github.com/miccunifi/FRED/blob/main/dataset_splits/challenging/challenging_test_split.txt) with [`src/phase1_fft.py`](../src/phase1_fft.py). It pooled them with the nine previous test clips and kept `train/36` separate because it was used for development. The FFT peak score threshold stayed at 5.2981 and the event count threshold at 4,578. The comparison covers the plain FFT and count scores on preselected boxes. The flicker filter, combined classifier and box finder were outside the scope of this run.

| Clips, busy backgrounds | Score | Precision | Recall | AUC | False alarms per background-box minute |
|---|---|---:|---:|---:|---:|
| 49 new | Plain FFT | 38.5% | 17.9% | 0.632 | 23.69 |
| 49 new | Event count | 50.3% | 51.2% | 0.789 | 42.04 |
| 58 pooled | Plain FFT | 32.9% | 15.9% | 0.615 | 26.97 |
| 58 pooled | Event count | 52.8% | 48.9% | 0.789 | 36.37 |

The pooled results cover 21,929 drone windows and 63,264 busy background windows. Event count ranked drone windows better across thresholds and caught more of them at the frozen threshold. It also produced more false alarms per minute at that threshold, so the recall comparison is at different false-alarm rates. Plain FFT reached 0.810 AUC against shifted boxes, mostly empty sky, compared with 0.615 against busy backgrounds.

See the [full report](../experiments/phase1_challenging_expansion/report.md), [run manifest](../experiments/phase1_challenging_expansion/manifest.json) and [progress log](../experiments/phase1_challenging_expansion/progress.log).

---

## November 6, 2026: steady-peak rejection cuts flicker; combined baseline frozen for the CfC

**Run:** `src/phase1_multiseq.py`, same sets, locked settings and 2% false-alarm rule as the entry below. Both additions were tuned on the tune sequences only.

**Steady-peak rejection (no mains frequency hardcoded).** For each window, look at the same box's other windows 0.5–4 s away (closer ones share samples). Stability is the fraction whose peak is within 2 Hz (one FFT bin) of this window's peak. If stability > 0.40, the window's FFT score is set to 0. The rule never looks at the frequency itself, so it works for 50 Hz or 60 Hz grids. FFT threshold re-tuned to 4.64.

- Active-box false alarms on a 50 or 60 Hz mains harmonic (counted only for this report) fell from 1977 to 214. test/111: 791 → 41; train/108: 578 → 0.
- Real drone windows rejected: 0.1% heldout, 0.2% challenging, worst DarwinFPV on challenging at 0.6%. No steady-hover problem showed up.
- Challenging Mini 3, the case the flicker broke, went from AUC 0.296 to 0.803 against active backgrounds.
- Caveat: the tune sequences contain almost no lights, so tuning measured the rule's cost, not its benefit. The 0.40 cut and the 4 s span are not validated against flicker.

**Combined baseline (the number the CfC has to beat).** Logistic regression in NumPy, fit on tune windows, on standardised log event count, log prominence, stability and log box area (the variant with the best leave-one-tune-sequence-out recall). Weights: log count +3.08, log prominence +1.27, stability +0.20, log area −0.61, bias −2.81; decision log-odds > 1.86. All weights, means and standard deviations are in `experiments/phase1_threshold.json`. The stability weight came out positive because the tune set has no flicker to learn from.

**AUC / recall / false alarms per minute, all types:**

| Set, background | FFT | FFT + stability | Count | Combined |
|---|---|---|---|---|
| Heldout, active | 0.886 / 0.537 / 7.3 | 0.891 / 0.595 / 7.7 | 0.932 / 0.637 / 10.3 | **0.952** / 0.652 / 8.7 |
| Heldout, shifted | 0.965 / 0.537 / 0.95 | 0.965 / 0.595 / 0.84 | 0.992 / 0.637 / 0.34 | **0.996** / 0.652 / 0.08 |
| Challenging, active | 0.527 / 0.059 / 44.0 | 0.706 / 0.072 / 5.0 | **0.797** / 0.374 / 6.9 | 0.775 / 0.226 / 10.2 |
| Challenging, shifted | 0.795 / 0.059 / 2.9 | 0.799 / 0.072 / 1.9 | 0.966 / 0.374 / 0.32 | **0.979** / 0.226 / 0.05 |

**What it means:**

- **The CfC's target:** heldout active AUC 0.952, challenging active AUC 0.775 (and count-only's 0.797 there, since the combined model loses to it). The CfC has to beat both on the same windows; the CSVs carry every score per window (`score_lr_stab_area` is the combined model).
- **The rotor frequency still doesn't beat event count on its own.** FFT + stability beats count only for Betafpv on heldout (0.765 vs 0.705) and Mini 3 on challenging (0.803 vs 0.738).
- **The combined model doesn't reject flicker.** It still has 187 mains-harmonic false alarms on train/35, because it never saw lights in training. Adding a sequence with lights to the tune set is the obvious fix, but it has to come from outside the challenging split.

**Files:** `experiments/phase1_multiseq_results.txt` (per type, precision and false-alarm columns), `experiments/phase1_threshold.json`, CSV columns `stability`, `score_fft_stab`, `score_lr_basic`, `score_lr_stab`, `score_lr_stab_area`. The `_ds2`/`_ds4` CSVs were not regenerated.

---

## November 6, 2026: finding the drone without FRED's boxes

**Run:** `src/phase1_find.py` (plain NumPy). Per 0.5 s window (0.25 s step):

1. Ignore hot pixels (pixels firing in ≥ 90% of the recording's 1 s blocks). This uses the whole recording, an offline shortcut; a live system would need a calibration clip.
2. Count events in 48 × 48 px cells. A cell is dense if its count is ≥ 16 × the window's mean cell count and ≥ 10.
3. Merge touching dense cells into clusters and keep only the largest one (K = 1). Its box stays fixed for the window.
4. Score that box with the locked FFT (prominence, 0.5 ms bins, 50–1000 Hz) and pass it if prominence > 3.25.

Cell size, the density factor, K and the threshold were picked together on the tune sequences to maximise F1 (tune F1 0.691), then frozen in `experiments/phase1_find_params.json`. A candidate finds the drone if its IoU (overlap area ÷ combined area) with the drone's box swept over the window is above 0.1.

**Results** (candidate recall = largest cluster lands on the drone; detection recall = it also passes the FFT; false candidates per minute before → after the FFT; precision of FFT-passing candidates):

| Set | Candidate recall | Detection recall | False / min | Precision |
|---|---|---|---|---|
| Heldout, all types | 0.765 | 0.696 | 53.4 → 21.2 | 0.882 |
| Challenging, all types | 0.428 | 0.237 | 71.3 → 22.9 | 0.689 |
| Development (train/36) | 0.516 | 0.396 | 49.8 → 11.1 | 0.923 |

Heldout per type (detection recall / false per min after FFT): Tello 0.955 / 8.2, Mini 3 0.843 / 18.3, Mini 2 0.761 / 1.6, DarwinFPV 0.621 / 20.7, Betafpv air75 0.219 / 57.4. Challenging: DarwinFPV 0.295, Mini 3 0.282, Betafpv 0.249, Tello 0.067.

Runtime: 0.038 s of compute per second of data overall, 0.134 s/s on the busiest sequence (train/199, 119 M events), file loading excluded. Fast enough for live use on a laptop; untested on a Raspberry Pi 5.

**What it means:**

- **Event density does most of the finding; the FFT adds little.** The frozen threshold of 3.25 sits just above the prominence of pure noise (about 3.0). The DarwinFPV passes at that level even though it has no rotor peak. Raising the threshold to 4 drops challenging detection recall to 0.092.
- **False candidates are other moving things.** Streaks that look like insects or birds and tree clutter pass with peaks around 100–200 Hz.
- **Noisy scenes break the density rule.** In test/178 and train/177 the Tello is plainly visible, but uniform sensor noise raises the window mean, so "16 × mean" misses it. A threshold of mean + k × standard deviation, or background subtraction per cell over time, might fix this. Not tried, to keep the challenging set untuned.
- **Clutter beats small drones.** With one candidate per window, the Betafpv air75 loses to clutter (150–180 false candidates per minute before the FFT).
- **One candidate per window can't report two drones.** K = 1 won only because every tune sequence has one drone, so it caps recall on train/35 and train/36.
- False counts are inflated where annotations have gaps, because a candidate on an unannotated drone counts as false.

**Next (decided November 6):** the box-free finder is the top priority for the live demo. Mains rejection must not hardcode 50 Hz harmonics: Canada's grid is 60 Hz (120/240/360 Hz flicker) and the Mini 2 sits near 300 Hz. Instead, reject peaks whose frequency stays fixed over time. Add a combined FFT + count baseline; that is the number the CfC has to beat.

**Files:** `experiments/phase1_find_results.txt` (per-sequence tables and runtime), `experiments/phase1_find_tune_log.txt`, `experiments/phase1_find_frames_*.png` (candidate boxes against ground truth; the test_178 one comes from a one-off diagnostic run), `experiments/phase1_find_prominence_*.png`.

---

## November 6, 2026: FFT loses to plain event counting on busy backgrounds

**Run:** `src/phase1_multiseq.py` over 20 FRED sequences, all five drone types. Settings locked at 0.5 ms bins, 50–1000 Hz, 0.5 s windows, 0.25 s step. Sets are in `experiments/phase1_sequences.json`:

- tune: train/91, 67, 96, 225, 218 (one per drone type, none in the challenging split)
- heldout: train/92, 58, 95, 224, 199
- challenging: test/111, train/108, test/21, train/78, test/178, train/177, train/99, train/100, train/35 (FRED has no single-drone Mini 2 sequence in the challenging split)
- development: train/36, reported separately because the detector was designed on it

Thresholds were frozen on the tune set as the lowest value with at most 2% false alarms on active background windows: FFT prominence > 5.30, event count > 4578 (`experiments/phase1_threshold.json`).

Two kinds of negatives:

- **shifted:** the drone's box moved to an empty part of the image, mostly sky.
- **active:** same-size boxes in the busiest 25% of the image (trees, buildings, lights), picked from event density every 5 s, preferring boxes whose event counts are closest to the drone's.

The count-only baseline scores a window by the number of events in the box and ignores timing.

**AUC, active backgrounds:**

| Set | FFT | Count only |
|---|---|---|
| Heldout, all types | 0.886 | 0.932 |
| Challenging, all types | 0.527 | 0.797 |
| Development (train/36), all types | 0.592 | 0.894 |

Per type, heldout (FFT / count): Betafpv 0.764 / 0.705, Mini 2 0.981 / 0.982, Mini 3 0.948 / 0.978, Tello 0.963 / 0.987, DarwinFPV 0.750 / 0.969. Challenging: Betafpv 0.684 / 0.777, Mini 3 0.296 / 0.738, Tello 0.501 / 0.955, DarwinFPV 0.668 / 0.794.

False alarms per minute of one background box at the frozen threshold, active, all types: heldout FFT 7.3 vs count 10.3; challenging FFT 44.0 vs count 6.9.

**What it means:**

- The teammate's bar (FFT must beat count-only on harder backgrounds) is **not met**. On this data, the event count alone detects drones better than the rotor frequency.
- **Light flicker** causes most FFT false alarms. Active-box peaks land at exactly 100, 200 and 300 Hz, consistent with lights on 50 Hz mains (FRED comes from the University of Florence, and Italy's grid runs at 50 Hz; the recording sites are not confirmed). In test/111, 705 of 1098 active windows peak at 200 Hz. The Mini 3's blade-pass frequency (200–230 Hz) sits next to the 200 Hz line, which is why the Mini 3 scores 0.296 there.
- **The count matching mostly failed.** In 18 of 22 drone tracks, the busiest background boxes had under half the drone's median event count (ratio 0.02–0.46). Where the ratio reached 0.46–0.62, the FFT beat count by a small margin (0.764 vs 0.705, 0.720 vs 0.703, 0.671 vs 0.658). So the count baseline's win partly reflects that nothing else in these scenes is as busy as a nearby drone.
- **Tello has no rotor signal** on the challenging sequences: FFT AUC 0.520 even against empty sky.
- **Box width:** FFT recall at the frozen threshold rises steadily with width: 0.07 (<20 px), 0.17 (20–30), 0.19 (30–45), 0.31 (45–70), 0.39 (70–100), 0.43 (100–150), 0.56 (>150). There is no sharp cutoff. Drone type matters more: Mini 2 reaches 0.64 at 20–30 px, Betafpv stays at 0.02–0.09 at every width.
- **The 2× and 4× downsampling test was a no-op** (all changes ≤ 0.005). Dividing coordinates only moves box edges; the score sums every event in the box, so the counts don't change. Simulating a smaller sensor needs a model of how each pixel generates events.

**Next:** reject light flicker (superseded: no fixed 100 Hz rule, see the finder entry above); find negatives as busy as the drone (birds, close moving objects) before trusting the count baseline.

**Files:** `experiments/phase1_multiseq_results.txt`, `experiments/phase1_width_bins.png`, `experiments/phase1_downsample.png`, per-window CSVs in `experiments/csv/`.

---

## November 6, 2026: seq 36 is in FRED's challenging test split

Every result below this entry was tuned on train/36, and train/36 appears in `dataset_splits/challenging/challenging_test_split.txt` in the FRED repo. From here on it is reported only as development data and never counted as challenging.

---

## November 6, 2026: DarwinFPV never hovers in seq 36; Mini 3 tone survives motion

**Run:** windows split by box-centre speed (0.1 ms bins).

- Mini 3: AUC 0.926 in fast windows vs 0.882 in slow ones (50–1000 Hz). Motion does not hide its rotor tone.
- DarwinFPV: only 3 of 361 windows move slower than one box-width per second, too few to test a hover. Its slowest third still shows no tone and few events (about 1,200 per window vs 6,200 for the slow Mini 3).

Superseded in part by the multi-sequence entry above (empty-sky negatives only).

---

## November 6, 2026: DarwinFPV event frames look like body motion

Event-frame crops (`experiments/phase1_d_frame_crops.png`): the Mini 3 shows speckle at four arm tips; the DarwinFPV shows one solid blob with opposite-polarity leading and trailing edges, the pattern a moving body makes. Each frame sums about 33 ms, so this is suggestive, not proof.

---

## November 6, 2026: DarwinFPV tone is not hiding above 1 kHz

**Run:** 0.1 ms bins, 50–5000 Hz. The DarwinFPV still shows no stable peak (8% of windows within ±10% of the median peak). Its old 558 Hz median is what pure noise gives in a 50–1000 Hz band, not an alias. The Mini 3 result is unchanged (AUC 0.911 at 50–1000 Hz).

---

## November 6, 2026: first FFT run, seq 36

**Run:** `src/phase1_fft.py` on train/36, 0.5 ms bins, 50–1000 Hz, negatives = empty-sky shifted boxes.

- DJI Mini 3: peak 228 Hz (IQR 206–264), second peak near 450 Hz. Precision 0.93 / recall 0.67 at threshold 4, AUC 0.91.
- DarwinFPV cineape20: no stable peak. AUC 0.78.

Superseded: tuned and tested on the same sequence, against empty sky only, and seq 36 is in the challenging split. See the multi-sequence entry.
