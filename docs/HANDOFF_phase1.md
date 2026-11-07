# Phase 1 handoff: FFT rotor detector

Written November 6, 2026. State of `main` after merging `phase1-fft-detector`.

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
