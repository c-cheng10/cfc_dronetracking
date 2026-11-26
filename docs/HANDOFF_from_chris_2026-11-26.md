# Handoff from Chris, November 26, 2026: finish the Phase 3 CfC sweep

I can't run anything for a while, so please finish this sweep on your machine. It's your step 2 from the CfC-lane handoff, unchanged as you asked: `time_mmcfc`, plain and drop-trained (0.9), at 4 learning rates on Phase 3, 5 seeds each, then `rate_shift` and `event_shift`. Training of the 6 surviving runs stopped at about 90% of their epochs. Every seed has a per-epoch checkpoint, and they're committed (force-added, about 0.7 MB) under `runs/phase3-cfc-2026-11-25/`, so `--resume` picks up where I stopped.

## 1. Where each run stopped

| Run | Seeds 0–3 | Seed 4 |
|---|---|---|
| lr 0.0003, plain | done (epoch 40) | epoch 25 of 40 |
| lr 0.0003, drop-trained | done | epoch 27 |
| lr 0.001, plain | done | epoch 24 |
| lr 0.001, drop-trained | done | epoch 28 |
| lr 0.003, plain | done | epoch 25 |
| lr 0.003, drop-trained | done | **NaN loss after epoch 15** (checkpoint is epoch 15) |
| lr 0.01, plain | **NaN loss after epoch 3 of seed 0**; seeds 1–4 never ran | — |
| lr 0.01, drop-trained | **NaN loss after epoch 12 of seed 0**; seeds 1–4 never ran | — |

## 2. Commands

Pull `chris` first. These need the Phase 3 recordings, because the packet caches now carry a version key (your reply, item 5) and your existing caches will rebuild once.

The checkpoint settings must match exactly or `--resume` refuses: `--threads 1 --script` (I used `--parallel 1`; `--parallel` isn't checked, so `--parallel 5` is fine). Seeds 0–3 load at epoch 40, skip training and only get scored; seed 4 carries on.

```bash
OUT=runs/phase3-cfc-2026-11-25
for lr in 0.0003 0.001 0.003; do
  python -m src.train.packets --arm time_mmcfc --split phase3 --lr $lr --threads 1 --parallel 5 --script --resume --out $OUT/lr$lr
  python -m src.train.packets --arm time_mmcfc --split phase3 --lr $lr --drop-augment 0.9 --threads 1 --parallel 5 --script --resume --out $OUT/lr$lr
done
for lr in 0.0003 0.001 0.003; do
  python -m src.train.rate_shift  --split phase3 --runs $OUT/lr$lr --out $OUT/lr$lr/rate_shift.json
  python -m src.train.event_shift --split phase3 --runs $OUT/lr$lr --out $OUT/lr$lr/event_shift.json
done
```

On my laptop an epoch took 120–170 s per process, so about 30–40 minutes of training remain, plus the tests.

Things to know:
- **Your machine finishes seed 4.** Your torch is 2.12.1 on macOS and mine is 2.15.0+cpu on Windows, so seed 4 won't be bit-identical to what my laptop would have produced. The CfC is sensitive to this (my seed 0 ranged 0.851–0.890 across thread settings on Phase 1). Please note in the results that seed 4 was finished on your machine.
- **lr 0.003 drop-trained** resumes from its epoch-15 checkpoint and will probably go NaN again. If it does, the whole run stops and writes no `results.json`. Its seeds 0–3 are scored in section 3, so report them plus the crash.
- **lr 0.01:** I wouldn't rerun it. Resuming from the same state should repeat the crash. To report all 5 seeds it needs a `--seeds` flag (not written), or we record "diverges at lr 0.01 on seed 0" as the result.
- **New since your reply:** `seed_N.json` bundles (item 6) are written by `packets.py` now. Run `python -m src.write_bundles <run dirs>` once on your own run folders to backfill them. On mine it backfilled `packets-2026-11-22` and `phase3-2026-11-24`, and the count statistics matched the window-CSV ones.

## 3. Results so far (Phase 3 validation, seeds 0–3)

Scored from the epoch-40 checkpoints with `predict` and `both_aucs`, exactly as `packets.py` does. They match the numbers the runs printed. The comparison is your Phase 3 models, 5 seeds.

| lr | mmCfC, matched AUC | GRU | LSTM | mmCfC drop-trained | GRU drop-trained |
|---|---|---|---|---|---|
| 0.0003 | 0.727 (0.549, 0.744, 0.745, 0.868) | 0.751 ± 0.091 | 0.763 ± 0.084 | 0.837 (0.831–0.843) | 0.848 ± 0.004 |
| 0.001 | 0.933 (0.918–0.944) | 0.951 ± 0.001 | 0.906 ± 0.016 | 0.914 (0.893–0.930) | 0.917 ± 0.009 |
| 0.003 | **0.955** (0.949, 0.954, 0.952, 0.962) | 0.969 ± 0.003 | 0.958 ± 0.003 | 0.940 (0.937–0.944) | 0.956 ± 0.004 |
| 0.01 | diverged | **0.972 ± 0.005** | 0.961 ± 0.007 | diverged | 0.961 ± 0.006 |

Plain AUC is level: mmCfC 0.979 mean at lr 0.003 (seeds 0–3), against your GRU's 0.979 at lr 0.01.

Read so far: the GRU is ahead at every rate, by about 0.015 matched AUC at each model's best setting (0.955 vs 0.972). At lr 0.003 the mmCfC is level with the LSTM and stable across seeds. The CfC's remaining case is robustness, which is what `rate_shift` and `event_shift` measure.

Per-seed AUC and matched AUC for the finished seeds:

```
lr0.0003/time_mmcfc       0.9506/0.7441  0.8961/0.5494  0.9517/0.7451  0.9643/0.8681
lr0.0003/time_mmcfc_aug   0.9564/0.8376  0.9530/0.8306  0.9556/0.8429  0.9504/0.8371
lr0.001/time_mmcfc        0.9752/0.9325  0.9744/0.9440  0.9784/0.9395  0.9702/0.9178
lr0.001/time_mmcfc_aug    0.9521/0.9210  0.9282/0.9301  0.9611/0.9108  0.9638/0.8929
lr0.003/time_mmcfc        0.9788/0.9544  0.9824/0.9619  0.9755/0.9493  0.9810/0.9524
lr0.003/time_mmcfc_aug    0.9573/0.9373  0.9606/0.9443  0.9653/0.9400  0.9746/0.9388
```

## 4. Why the CfC crashes: exploding gradients

I replayed epoch 4 of lr 0.01 seed 0 from its epoch-3 checkpoint, logging the loss and gradient norm (before clipping) every batch:
- Batches 0–208 were normal: gradient norm about 1 (median 1.02), loss falling.
- From batch 209 the norm jumped to 83, then 947, 2,300 and 5,008 over the next 30 batches, and the loss went from 0.4 to 1.2.
- It settled by batch 250.
- My replay used 4 threads, so it isn't bit-identical to the run, and it survived. In the real run one of these bursts reached infinity. `clip_grad_norm_` can't scale an infinite norm, so the weights went NaN.
- The weights were small right before every crash (largest |w| about 2), so this is a sudden burst, not slow drift.

The cause: backpropagation runs through up to 500 sequential CfC steps, and the CfC rebuilds its state every step. The GRU's gated memory keeps that chain near 1, and the CfC has nothing equivalent. Higher learning rates reach the burst-prone regions sooner, which fits lr 0.01 dying first, lr 0.003 late, and the slower rates surviving. Bursts that don't reach infinity probably also explain the seed spread (lr 0.0003: 0.55 to 0.87).

## 5. Proposal, for us to agree before anything runs

1. **Skip a step whose gradient norm isn't finite.** Check the value `clip_grad_norm_` returns, and if it isn't finite, `zero_grad` and skip `optimizer.step()`. One line. It only acts when a step would already destroy the model, so it can't change any run that never hits inf. It applies to every model, per the rules.
2. **Your warm-up plus cosine decay,** tested on Phase 1 for every model, adopted only if the GRU doesn't lose. As you proposed.
3. Clip 0.5 and shorter windows (fewer packets per window) only if 1 and 2 aren't enough.

If you agree, I'll put 1 and 2 on a new branch off `chris`, so this sweep stays the "before" result.

## 6. Also
- `chris` was force-pushed today to remove an attribution line from 5 commit messages. The code is identical (same tree). If you'd fetched `chris` before, reset to `origin/chris`.
- The open PR is still `max ← chris`: https://github.com/c-cheng10/cfc_dronetracking/compare/max...chris

Chris
