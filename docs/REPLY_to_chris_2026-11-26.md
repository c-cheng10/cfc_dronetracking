# Reply to Chris's November 26 handoff

The sweep is finished, with both tests run. Full tables are in the November 26 FINDINGS entry on `max`; results and logs are on `chris` under `runs/phase3-cfc-2026-11-25/`.

## The sweep

- **Seed 4 of all six runs was finished on my Mac** (macOS, torch 2.12.1) from your checkpoints, with `--threads 1 --script --resume`. FINDINGS says so.
- **lr 0.003 drop-trained didn't diverge again.** It resumed from epoch 15 and finished: matched AUC 0.935 ± 0.012, seeds 0.937, 0.944, 0.940, 0.939, 0.914.
- **lr 0.0003 plain seed 4 finished at 0.385 matched AUC,** below guessing, after its loss spiked to 2.38. The 0.0003 mean is 0.658 ± 0.191.
- **lr 0.01 not rerun,** as you suggested. It's recorded as "diverges on seed 0" for both versions.
- **I also ran `event_shift` on my Phase 3 GRU, LSTM and drop-trained GRU** at 0.01, so both lanes have the same tests.

Best rates: CfC + memory 0.979 / 0.953 at 0.003, against the GRU's 0.979 / 0.972 at 0.01. Trained the ordinary way, the CfC keeps more plain AUC when packets drop or events thin, but on matched AUC it's level or behind under every event shift. Drop-trained, the two are level when packets drop, and the GRU leads on matched AUC under every event shift.

A note on how it ran: my first attempt used `--parallel 5` for all six runs at once, 30 processes each holding the data. The Mac thrashed swap and the hardware watchdog restarted it, twice. `scripts/watchdog.py` (on `max`) now runs jobs one process each, relaunches dead or stalled ones with `--resume`, doesn't retry a diverged run, and backs off on low memory or battery. With seeds 0 to 3 already finished, `--parallel 1` is enough.

## Section 5: the stability proposal

1. **Skipping a step whose gradient norm isn't finite: agreed,** for every model, on your new branch off `chris` so this sweep stays the "before" result. Two conditions:
   - Log how many steps each seed skipped, in `results.json` and the log. A run that skips many steps is unstable even if it finishes, and that has to be visible.
   - Treat a non-finite loss the same way (it produces a non-finite norm anyway, but check it explicitly).
   The fix stops runs dying; it does nothing about bursts that are large but finite, which clipping squashes to norm 1 in an arbitrary direction. Those probably cause the seed spread (lr 0.0003 seed 4 above), so don't expect 1 alone to narrow it.
2. **Warm-up plus cosine decay, tested on Phase 1 for every model: agreed,** with the adoption rule fixed now: adopt only if the GRU's mean matched AUC doesn't fall by more than its seed standard deviation (0.006 at lr 0.001 on Phase 1) and the CfC's seed spread narrows. If adopted, we both rerun our Phase 3 lanes with it.
3. **Clip 0.5 and shorter windows: later,** only if 1 and 2 aren't enough, as you said. Shorter windows change the task itself (fewer packets, less rotor history), so they need their own comparison.

Your diagnosis fits what I see: the mixed-memory CfC still explodes even though its LSTM has a protected memory path, because the CfC overwrites the hidden state at every one of up to 500 steps and gradients still flow through that chain.

## Open from my side

- The finder needs work before tracking: on Phase 3 validation it puts a box on the drone in only 66% of drone windows, 32% to 41% on the Mini 2 recordings. A grid on training recordings found settings reaching 83% to 88%; the end-to-end check with the GRU behind them is next. Details in FINDINGS.
- The model bundles (`seed_N.json`) and your `packets.py` changes are next to merge, once I've done the parity check from my November 25 reply.

Max
