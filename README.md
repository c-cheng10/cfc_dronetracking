# cfc_dronetracking

Event-camera drone detection with liquid neural networks (CfC), built toward the Canadian IDEaS counter-drone work.

**Core question:** can a small CfC confirm a drone from its rotor flicker in an event stream, and run fast enough for a Raspberry Pi 5? Can it beat (a) a plain FFT frequency baseline and (b) a GRU given Δt, at the same parameter count?

## Layout

```
src/data/      loading event streams, cutting patches, binning events in time
src/models/    FFT baseline, GRU, CfC (ncps first, then our own cell)
src/train/     training and evaluation scripts
experiments/   one markdown log per run: config, seeds, results
notebooks/     exploration and plots
data/          datasets (git-ignored)
```

## Plan (timeline set 2026-11-01)

Target: **IDEaS Counter-UAS (Urban) Sandbox 2027**. Applications close **1 Dec 2026, 2:00 PM EDT** through the Defence Innovation Portal. The application needs TRL 5, or TRL 4 plus a plan to reach TRL 5 by **10 Mar 2027**, and a real (not simulated) demo video of up to 3 minutes. The sandbox runs in Ottawa, late July 2027.

| Phase | When | Goal | Done when |
|---|---|---|---|
| 0 | Week of Nov 2 | Confirm eligibility (Canadian citizen; still confirm the sandbox's applicant rules); order the camera (quote forms sent to CenturyArks and Prophesee, awaiting replies as of 2026-11-06); measure the Mini 4K's blade-pass frequency (phone spectrogram); CfC sine test ✅ | Eligibility answer in writing; camera shipped |
| 1 | Nov 2–20 | FRED: plot events, then an **FFT rotor-frequency detector** (NumPy only, no ML) | Precision/recall on FRED train and the challenging split |
| 2 | Nov 15–30 | **Dec 1 sprint:** live camera + FFT detector on the Mini 4K, a 3-min unedited video, TRL 4 application | Application submitted (or a deliberate no-go by Nov 22) |
| 3 | Dec–Jan | ML baselines on FRED patches: FFT + logistic regression, GRU+Δt, `ncps` CfC, 5 seeds, matched params; then test on SkyEV / held-out data | Comparison table |
| 4 | Jan–Feb | **Our own CfC cell**: batched variable Δt, quantization, Raspberry Pi 5 latency | Matches `ncps`; ms per patch on a Pi 5 |
| 5 | Feb–Mar 10 | Field tests: hover at 20 / 50 m, passes at 5 / 10 / 16 m/s; our own Mini 4K dataset; fine-tune | TRL 5 evidence by Mar 10 |
| 6 | Mar–Jul | Science fair (if chosen), sandbox prep | |

## Rules

- No single-seed results. Report the mean ± standard deviation over at least 5 seeds.
- Every model compared at the same parameter count.
- Never touch the test split until the final numbers.
- **Bin width ≤ 1 / (4 × rotor blade-pass frequency).** Measure the Mini 4K's frequency with a phone spectrogram first.

## Data

- **FRED** (Florence RGB-Event Drone dataset, arXiv 2506.05163): 7+ h, 5 drone types. One zip per sequence: raw events in `.hdf5`, boxes in `coordinates.txt`. Code: `github.com/miccunifi/FRED`. Put it in `data/FRED/`. Raw events (`Event/events.hdf5`, dataset `CD/events`, fields x, y, p, t in µs) use Prophesee's ECF compression (HDF5 filter 36559), so plain h5py can't read them. Convert each sequence once with `python -m src.data.ecf_convert <path to events.hdf5>`, which writes `events.npy` beside it, then open that with `np.load(path, mmap_mode="r")`. The decoder is in `third_party/ecf/`; see its README to build `ecf.dll` on a new machine. The bundled PNG event frames are about 30 fps, far too slow for rotor flicker.
- **SkyEV** (ETH, 2026): held-out generalization test.
