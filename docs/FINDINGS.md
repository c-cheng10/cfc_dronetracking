# Findings log

A running record of what we have measured, newest first. Add a dated entry for each result, including the ones that go against us, and never edit an old entry to change what it said: if a later result overturns it, add a new entry and mark the old one superseded.

Each entry says what was run, the numbers, and where the files are. AUC is the probability that a random drone window scores higher than a random background window (0.5 = coin flip, 1.0 = perfect).

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

**Next:** reject peaks near multiples of 100 Hz (tuned on the tune set only) and re-test; find negatives as busy as the drone (birds, close moving objects) before trusting the count baseline.

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
