# Phase 1 handoff: FFT rotor detector

Written November 6, 2026. State of `main` after merging `phase1-fft-detector`.

**Superseded results:** the numbers below come from train/36 alone, tuned on itself, against empty-sky negatives. The current results (20 sequences, frozen threshold, busy backgrounds, count-only baseline) are in `docs/FINDINGS.md`. The run commands have also changed: use `python src/phase1_fft.py --seq train/36` for one sequence and `python src/phase1_multiseq.py` for the full evaluation.

## What Phase 1 tests

A spinning propeller blade sweeps the same pixels many times a second, so the number of events inside a drone's bounding box rises and falls at the blade-pass frequency (revolutions per second × blades). `src/phase1_fft.py` counts the events inside each drone box in fixed time bins, runs an FFT on every 0.5 s window (0.25 s step, Hann taper), and scores each window by prominence: the tallest peak divided by the median magnitude in the band. It does the same for background boxes of the same size, shifted away from any drone. Drone windows are positives, background windows are negatives, and a threshold on prominence gives precision and recall. No machine learning.

## How to run it (Windows)

1. Download sequence 36 and unzip it so the events sit at `data\FRED\train\36\36\Event\events.hdf5`:

   ```
   .venv\Scripts\python -c "from huggingface_hub import hf_hub_download as d; print(d('Nyquispei/FRED', 'train/36.zip', repo_type='dataset', local_dir='data/FRED'))"
   ```

   The zip is 314 MB and unpacks to 383 MB. Extract it into `data\FRED\train\36\`.

2. Convert the events once with your converter (needs `third_party\ecf\ecf.dll`, see `third_party\ecf\README.md`):

   ```
   .venv\Scripts\python -m src.data.ecf_convert data\FRED\train\36\36\Event\events.hdf5
   ```

3. Run the detector from the repo root. It takes about 22 s on Max's Mac.

   ```
   .venv\Scripts\python src\phase1_fft.py
   .venv\Scripts\python src\phase1_fft.py --bin-ms 0.1 --f-hi 5000
   .venv\Scripts\python src\phase1_fft.py --bin-ms 0.1 --f-hi 5000 --speed-split
   ```

   Any non-default option adds a suffix to the output names, so runs never overwrite each other. Everything lands in `experiments\`.

## Results on sequence 36

Sequence 36 has two drones: ID 1, a DJI Mini 3 (open propellers), and ID 2, a DarwinFPV cineape20 (a 2" cinewhoop with ducted propellers).

| | DJI Mini 3 | DarwinFPV |
|---|---|---|
| Peak frequency, median (IQR) | 228 Hz (206–264) | no stable peak |
| Precision / recall at threshold 4 | 0.93 / 0.67 | 0.86 / 0.09 |
| AUC | 0.91 | 0.78 |

AUC is the probability that a random drone window scores higher than a random background window. Default run: 0.5 ms bins, 50–1000 Hz.

- The Mini 3 also shows a weaker peak near 450 Hz, about double the main one.
- Its tone is stronger when it moves fast (AUC 0.93 fast vs 0.88 slow, 0.1 ms bins, 50–1000 Hz), so motion does not hide it.
- The FPV shows no rotor peak anywhere up to 5 kHz (0.1 ms bins), so its tone is not just above the old 1 kHz limit. Its AUC above 0.5 tracks its speed, which points to body motion rather than rotors.
- In the event frames (`experiments\phase1_d_frame_crops.png`) the Mini 3 looks like speckle at four arm tips and the FPV looks like one solid blob with opposite-polarity leading and trailing edges.

Full tables are in `experiments\phase1_fft_seq36_results*.txt`.

## What is not settled

- **Weak negatives.** The background boxes are mostly empty sky (about 0.01–0.03 events per 0.1 ms bin), so these numbers test "rotor vs nothing". Trees, birds and flickering lights are untested, and precision there will be lower.
- **Why the FPV is invisible.** Ducts hiding the blades and the rotors being too small in the image both fit. The FPV never hovers in this sequence (3 of 361 windows move slower than one box-width per second), so the hover test could not decide.
- **One sequence only.** The README's Phase 1 bar is precision and recall on FRED train and the challenging split. We have one train sequence.
- **Annotation gap.** The Mini 3 has no boxes from about 65 to 93 s, which removes 191 of its 453 windows.
- The bin-width rule in the README (≤ 1 / (4 × blade-pass)) puts 0.5 ms bins at a 500 Hz limit. The Mini 3 at about 228 Hz is inside it; the FPV would need 0.1 ms or finer.

## Next steps, in order

1. Background boxes on cluttered regions (trees, buildings, lights) so precision means something.
2. More FRED sequences, including other drone types and the challenging split.
3. A close hover of an FPV or our Mini 4K, to settle whether ducted drones are detectable at all.
4. Measure the Mini 4K's blade-pass frequency with a phone spectrogram, which Phase 0 still lists.

## Answers to your questions

### FFT plots: one drone patch and one background patch

`experiments/phase1_e_single_window_example.png` shows one 0.5 s window per drone, drone box against a background box at the same moment. I took the window whose prominence is closest to that drone's median, so it is a typical window, not the best one. Regenerate it with `python src/phase1_example_window.py`.

- Mini 3 (t = 2.00 s): a clear spike near 228 Hz, prominence 4.9 against 2.8 for the background.
- FPV (t = 57.50 s): no spike stands out, prominence 3.1 against 3.0.

In both, the background's raw magnitude is about a tenth of the drone's, because the background box is empty sky with almost no events. Prominence divides by each window's own median, which removes that gap, but a detector that only counted events would also beat this background. That is why the precision below is optimistic.

The median spectra over all windows are in `phase1_b_spectrum_drone_vs_control*.png`.

### Precision, recall, recordings, tuning, challenging split

- At prominence threshold 4 (0.5 ms bins, 50–1000 Hz): Mini 3 precision 0.93, recall 0.67; FPV precision 0.86, recall 0.09.
- Tested on one recording only: FRED `train/36`.
- Yes, tuned on that same recording. I picked threshold 4, the bin widths and the bands after looking at its results, so every number here is in-sample.
- No, the challenging split has not been run.

### Does it find the drone by itself?

No. It uses FRED's boxes from `coordinates.txt`, and it only answers "is there a rotor inside this box?". The background boxes are the drone's box shifted by up to 450 px horizontally and 250 px vertically, valid only where they stay inside the image and at least 10 px from every drone. A real detector would need a sliding window or a region proposal step in front of it.

### Frequency range, threshold, frequencies found

- Default band 50–1000 Hz with 0.5 ms bins. Also run at 50–5000 Hz with 0.1 ms bins.
- Score: prominence, the tallest peak divided by the median magnitude in the band. Thresholds 2 to 15 are in the results files; 4 is the one quoted.
- Mini 3: peak median 228 Hz (IQR 206–264 Hz), plus a weaker peak near 450 Hz.
- FPV: no stable peak in either band. Its window peaks scatter across the whole band (median 558 Hz at 50–1000, 2120 Hz at 50–5000), which matches pure noise, whose median would sit near the band centre.

### False alarms on background

At threshold 4: 14 of 735 background windows (1.9%) beside the Mini 3, and 5 of 1002 (0.5%) beside the FPV. Windows are 0.5 s long and overlap by half. These are empty-sky boxes, so expect more false alarms on trees, birds or lights.

### Input and output

Input:
- `events.npy` from `src/data/ecf_convert.py` (fields x, y, p, t; t in µs).
- `coordinates.txt` for the boxes.

The script counts events per box in fixed bins and cuts them into 0.5 s windows with a 0.25 s step. For each window it computes the start time, peak frequency, prominence, and whether the box was valid for the whole window. The decision is `prominence > threshold`.

Output today is only text summaries and plots in `experiments/`. It does not yet save one row per window, so it can't be compared row by row with your model. A proposal for both of us, one CSV row per window:

| column | meaning |
|---|---|
| `sequence` | FRED sequence, e.g. `train/36` |
| `track_id` | drone ID from `coordinates.txt` |
| `is_background` | 1 for a shifted background box |
| `dx`, `dy` | background shift in px (0, 0 for the drone box) |
| `t_start_us`, `t_end_us` | window edges, µs |
| `score` | model output, higher means more drone-like |
| `label` | 1 = drone inside, 0 = no drone |

Same boxes and windows for both models means precision, recall and AUC compare directly. Tell me if you want a different window length before I add this to the script.

### What doesn't work yet

- **Ducted drones.** The FPV is not detected; see "What is not settled".
- **Finding the drone.** It needs FRED's boxes.
- **Clutter.** Untested; the negatives are empty sky.
- **Far away.** Untested directly. The FPV box is about 47 px wide against 99 px for the Mini 3, so small size may be part of why the FPV fails.
- **Fast.** Not a problem for the Mini 3: its AUC is 0.93 in fast windows and 0.88 in slow ones.
- **Turning.** Not tested; the script does not measure turns.
