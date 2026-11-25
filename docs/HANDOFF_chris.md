# Handoff for Chris: the CfC lane

Written November 24, 2026, on the `max` branch. Max runs the GRU, LSTM, CNN and baselines; you run the CfC. The comparison between them is the result, so the two of us share one setup and neither changes it alone. The section on what must be the same is the one to read twice.

Background: [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) explains the models and results in plain terms, [CFC_EXPERIMENT.md](CFC_EXPERIMENT.md) the design, [FINDINGS.md](FINDINGS.md) every result with its caveats, newest first, and [HANDOFF_cfc.md](HANDOFF_cfc.md) the earlier technical handoff.

## Where it stands

On validation, at each model's best learning rate, the GRU scores 0.990 AUC and 0.971 matched AUC. The best CfC trained on complete windows (CfC + memory, `time_mmcfc`) scores 0.967 and 0.889. A CfC degrades more gracefully than a normally trained GRU when packets go missing, but a GRU trained with packets dropped matches it. Both CfC variants trained on complete windows diverged one learning rate above their best, while the GRU kept improving up to 0.01.

Those numbers come from 5 training and 5 validation recordings. From now on we use the Phase 3 split (`--split phase3`): 22 training recordings with 40,314 windows (10,629 drone) and 15 validation recordings with 21,114 windows (5,370 drone). Validation has 1,904 matched pairs, and no recording supplies more than 21% of them (the old split had 663 pairs, 54% from one recording). Training drone windows by type: Tello 3,310, Mini 2 2,866, Betafpv 1,780, DarwinFPV 1,772, Mini 3 901. Max's lane on this split (November 25, see FINDINGS): GRU 0.979 AUC / 0.972 matched AUC at rate 0.01, LSTM 0.979 / 0.961, drop-trained GRU 0.973 / 0.961, CNN 0.957 / 0.881. A GRU trained on complete windows falls to 0.549 AUC with half the packets dropped; the drop-trained GRU keeps 0.966. Those are the numbers the CfC is compared with.

## Step 1: get the same data

1. Clone `https://github.com/c-cheng10/cfc_dronetracking` and check out `max`. Make your own branch from it, for example `chris`, and send changes back as pull requests into `max`.
2. Make a Python environment with `numpy`, `torch`, `h5py`, `huggingface_hub` and `matplotlib` (versions in `requirements.txt`; torch was 2.12.1 on Max's Mac). No GPU is needed.
3. Build the ECF decoder, which FRED's compressed event files need:
   - macOS or Linux: build `third_party/hdf5_ecf` as its README says, then run every data command with `HDF5_PLUGIN_PATH=third_party/build`.
   - Windows: build `third_party/ecf/ecf.dll` as `third_party/ecf/README.md` says. The fetch script falls back to it automatically.
4. Download everything:

   ```
   HDF5_PLUGIN_PATH=third_party/build python -m src.data.fetch_fred
   ```

   It downloads 47 recordings (22 training, 15 validation, 9 test, 1 development) at the pinned Hugging Face revision, about 34 GB. It decodes each into `events.npy`, deletes the zip and the compressed file, and builds any missing window CSV. Leave room for the downloads plus the decoded files while it runs.
5. **Check the data matches.** The script compares the SHA-256 of every `events.npy` and window CSV with `experiments/phase3_checksums.json`, which Max's machine wrote, and stops with `MISMATCH` if one differs. Don't train until it finishes with no failures. If something mismatches, tell Max before changing anything.

## Step 2: check your machine reproduces Max's numbers

Train the existing mixed-memory CfC on the old split at its best rate:

```
python -m src.train.packets --arm time_mmcfc --split phase1 --lr 0.001 --out runs/check
```

Max's mean validation matched AUC is 0.889 ± 0.015 over 5 seeds (`runs/packets-2026-11-22/time_mmcfc/results.json`). A different CPU or torch build can shift individual seeds slightly, but your mean should land within about one standard deviation. If it doesn't, stop and compare with Max.

## What must be the same for both of us

Change any of these only after you've both agreed, then rerun every model on both sides and add a FINDINGS entry saying what changed.

**Data**
- The recordings in `experiments/phase3_sequences.json`: 22 training, 15 validation, the test set and the development recording. Never edit the file; `src/data/phase3_split.py` refuses to overwrite it.
- The data itself, checked by `experiments/phase3_checksums.json`.
- The window CSVs in `experiments/csv/`: which boxes and windows exist, and their labels. They come from `src/phase1_fft.py` and are in git.

**Inputs**
- Packets from `src/data/event_packets.py`: at most 500 per window, with the columns and the `log1p` of the gap exactly as `src/train/packets.py` builds them.
- "Time" features only (gap and ON fraction) for any headline result. The position and spread columns leak how boxes were placed.
- Event count joined at the readout, standardised with training statistics.
- Packet dropping from `src/data/packet_drop.py`, including `--drop-augment 0.9` for drop-trained models and the test rates 0, 25, 50, 75 and 90% in `src/train/rate_shift.py`.

**Training**
- About 4,900 parameters, within 2%. Check with `sum(p.numel() for p in model.parameters())`.
- 40 epochs, batch 128, AdamW, gradient norm clipped at 1, the loss weighted for class balance, the last epoch kept (no picking the best epoch).
- Seeds 0 to 4, always all five, and report the mean ± standard deviation.
- Learning rate chosen from 0.0003, 0.001, 0.003 and 0.01 by the best mean validation matched AUC. Rates outside that set have to be tried for every model.

**Evaluation**
- AUC and matched AUC, with the matched pairs from `src/count_matched.py` (tolerance 0.15 in `log1p` count, seed 0).
- The alarm threshold: the value that lets 2% of training background windows through.
- Paired comparisons by seed ("ahead in 5 of 5 seeds"), not just means.
- **The test set stays closed.** It's FRED's challenging split and the only clean test left, because validation is now used to choose learning rates. We open it once, together, when both lanes are finished.

## What you can change freely

- **The CfC cell and anything that is part of the CfC:** its wiring, the gate, how Δt enters it, the memory path, the backbone. Add each version as a new entry in `PACKET_ARMS` in `src/models/packet_models.py` with a new name, rather than changing an existing arm, so old results stay reproducible.
- **CfC-only checks,** such as comparing the cell with the reference `ncps` code at batch size 1.

Anything that changes how every model trains (a warm-up schedule, a learning-rate decay, a different clip value, a new augmentation) isn't CfC-only. Propose it, and if we agree, run it for every model.

## What to try first

1. **Train stability.** Both CfC variants diverge at fast learning rates the GRU handles. A warm-up schedule or tighter clipping might fix that; it counts as a training change, so test it for every model.
2. **The CfC at its best on the new split:** `time_mmcfc` and the drop-trained version, all four learning rates, `--split phase3`.
3. **Where the CfC should win:** a real change in event rate, from Max's SilkyEvCam recordings once they exist, instead of random packet drops.

## The deployment target

The drone we must detect is Max's DJI Mini 4K, and FRED has none. The new split leans towards the closest FRED types, the DJI Mini 2 (8 training recordings) and Mini 3 (every eligible recording). Real Mini 4K recordings with our camera come next; until then, no result here says how well we detect the Mini 4K.

## Known weaknesses

- Neighbouring FRED recordings, such as Mini 2 recordings 202 to 216, are split between training and validation and may share a location. That would make validation look better than a new site would.
- No birds in any training or validation data. EV-Flying has birds, but has to be requested from the FRED lab.
- The boxes come from FRED's labels. A real detector has to find them first.
