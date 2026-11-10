# Findings log

A running record of what we have measured, newest first. Add a dated entry for each result, including the ones that go against us, and never edit an old entry to change what it said: if a later result overturns it, add a new entry and mark the old one superseded.

Each entry says what was run, the numbers, and where the files are. AUC is the probability that a random drone window scores higher than a random background window (0.5 = coin flip, 1.0 = perfect).

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
