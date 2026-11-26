# Reply to Chris's handoff, November 25, 2026

Answers to your seven questions, in your order. Thanks for the parity work. Matching checksums across both FRED mirrors is a relief.

## 1. EventDetector score

Validation AP50 **0.524 ± 0.016** and AP75 **0.057 ± 0.011**, 5 seeds, 8 epochs, from `runs/fred-detector-2026-11-12/results.json`. It's on the old (Phase 1) split. You're right that nothing records it; I'll add a FINDINGS entry. The weights aren't in git yet. Say if you want them pushed, or retrain it with `python -m src.train.event_detector --out runs/fred-detector` (it uses `load_splits()`, so the Phase 1 split).

AP75 of 0.057 means its boxes are rarely tight. It finds roughly where a drone is but not its outline, which matters if the GRU behind it expects FRED-tight boxes.

## 2. Commit `1c50854`

The message is wrong, and it's my mistake. The detector trained on the 5 Phase 1 **training** recordings (train/91, 67, 96, 225, 218) and was scored on the 5 validation recordings (train/92, 58, 95, 224, 199). `src/train/event_detector.py` takes them from `load_splits()`. I meant "scored on validation only, not the challenging split". Its validation score is usable. It would still need retraining on the Phase 3 split before it feeds a tracker, so that it matches the GRU.

## 3. Reproduction with the GRU

Already done on my side: `runs/packets-2026-11-22/time_gru/` is exactly `--arm time_gru --split phase1 --lr 0.001`. Run it on yours and compare per seed:

| Seed | 0 | 1 | 2 | 3 | 4 | Mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| Matched AUC | 0.908 | 0.915 | 0.914 | 0.916 | 0.902 | 0.911 ± 0.006 |
| AUC | 0.980 | 0.978 | 0.981 | 0.978 | 0.980 | 0.979 ± 0.001 |
| Final loss | 0.341 | 0.346 | 0.339 | 0.313 | 0.338 | |

Its worst loss after epoch 5 was 0.754 (seed 1), and every seed recovered. Mine ran with the old `packets.py`, eager mode, 2 threads per process, torch 2.12.1 on macOS. If your GRU lands within about 0.01 of these, the shared setup is fine and the CfC gap is the CfC's own instability.

My mixed-memory CfC's final losses were 0.446, 0.442, 0.383, 0.386 and 0.380, so seeds 0 and 1 on my machine also ended above the 0.39 you call good.

## 4. Stability fix

My suggestion, for us to agree before anything runs:

1. **Run the Phase 3 CfC sweep now, unchanged.** My GRU, LSTM and CNN Phase 3 runs used the unchanged protocol, so your sweep is directly comparable to them. Report the instability as a result.
2. **Report the median and every seed beside the mean,** since one stumbling seed can move a mean by 0.04.
3. **Then test one combined change on Phase 1 for every model:** linear warm-up over the first epoch plus cosine decay to zero, clip left at 1. The spikes we've seen came mid-training (my count-window GRU's seed 0 spiked to 1.195 around epoch 16, your CfC seed 4 at epoch 25), which warm-up alone wouldn't touch, while decay lowers the step size late on. A clip of 0.5 changes how every model learns even when nothing spikes, so I'd leave it out unless decay isn't enough.
4. **Adopt it only if it doesn't hurt the GRU.** If we adopt it, we both rerun our Phase 3 lanes with it.

If you'd rather the sweep wait for the fix, that's fine too: it costs one more day.

## 5. Merging the shared-script changes

Yes, through a pull request, once one check passes on my machine: I rerun one existing arm with `--parallel 5` and confirm it matches `runs/packets-2026-11-22` to the same tolerance you saw. The `no_grad` change is fine. After the merge I'll rerun `rate_shift.py` and `sparsity_slice.py` once to confirm they still match.

For the packet cache with no staleness key: please add a key (a version number in `event_packets.py`, like the window cache) in the same PR, so nobody has to remember to delete it by hand.

## 6. Model bundle

Yes. Save `count_mean`, `count_std`, the alarm threshold, the arm name, feature set and learning rate beside each seed's weights, for example as `seed_N.json`. For the models already trained, compute those once from their training recordings and write the same file, so everything loads the same way.

## 7. Track-level metrics

Agreed: share of drones with a confirmed track, seconds to first confirmation, false confirmed tracks per minute, and ID switches. Three additions, all fixed before building:

- **False tracks per minute of camera time,** not per background-box minute, since that's what an operator sees.
- **The match rule:** when a track counts as on a drone (an IoU threshold against FRED's box, held for most of the track's life), and how long a lost track may coast before it ends.
- **The confirmation rule** (for example 3 of the last 4 windows), fixed on the training two-drone recordings and then left alone.

Your proposed order makes sense. For step 1: the saved validation rows carry `track_id` (it's a column in every `experiments/csv/*.csv`), so voting by recording and track ID works on my saved Phase 3 scores today.

## Also

- **Event thinning:** agreed, it's the more realistic shift. If we use it as augmentation, it's a training change, so it goes to every model.
- **`event_shift.py --split phase3`:** please add it; I'll run it on my Phase 3 GRU, LSTM and drop-trained GRU.

Max
