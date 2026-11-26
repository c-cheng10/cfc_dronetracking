"""Does the GRU still work on boxes the finder proposes, instead of FRED's hand-drawn ones?

python -m src.finder_gru --out FILE

For every Phase 3 validation recording, src/phase1_find.py proposes up to
MAX_CANDS boxes per 0.5 s window (frozen Phase 1 parameters, except max_cands
raised from 1). Each box's events are packetized exactly as for training and
scored by Max's Phase 3 GRUs at rate 0.01, 5 seeds, each at its own threshold
(2% of training background windows pass). The locked FFT score on the same
boxes is the comparison.

A drone counts as present in a window when it has at least 12 annotation frames
in it, and a box matches it when IoU with its swept box is above 0.1, both as in
phase1_find.py. Reported:
  found      share of present drones matched by any proposed box (the finder alone)
  detected   share of present drones matched by a box that passes the scorer
  false/min  passing boxes that match no drone, per minute of full frame
  box AUC    the scorer's AUC, boxes that match a drone against boxes that don't

Claim fixed before any result was seen (2026-11-25): the drop-trained GRU detects
at least 90% of the drones the finder's boxes find, and gives fewer false alarms
per minute than the FFT on the same boxes.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src import phase1_find as finder
from src import phase1_fft as phase1
from src.data.cnn_windows import load_splits
from src.data.event_packets import FEATURES, packets_from_events
from src.models.packet_models import PACKET_ARMS
from src.track_vote import ARMS, RUNS, scored_models
from src.train.packets import predict, tensors

MAX_CANDS = 3


def candidates(sequence, prm):
    """run_sequence's candidates and present drones, plus each candidate's packets and log count."""
    ev = np.load(phase1.events_path(sequence), mmap_mode="r")
    x, y, t = (np.asarray(ev[k]).astype(dt) for k, dt in (("x", np.int32), ("y", np.int32), ("t", np.int64)))
    result = finder.run_sequence(sequence, prm, ev=(x, y, t))
    p = np.asarray(ev["p"]).astype(np.float32)
    packets, log_count = [], []
    for w, x1, y1, x2, y2, *_ in result["cands"]:
        w0 = int(result["win_starts"][int(w)])
        i0, i1 = np.searchsorted(t, [w0, w0 + finder.WIN_US])
        xs, ys = x[i0:i1], y[i0:i1]
        inside = np.flatnonzero((xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)) + i0
        width, height = max(x2 - x1, 1), max(y2 - y1, 1)
        packets.append(packets_from_events((t[inside] - w0) / 1000.0, p[inside],
                                           ((x[inside] - x1) / width).astype(np.float32),
                                           ((y[inside] - y1) / height).astype(np.float32)))
        log_count.append(np.log1p(len(inside)))
    return result, packets, np.array(log_count, dtype=np.float32)


def summarise(results, scores, threshold, fft_threshold):
    """Pooled metrics for one scorer over all recordings. scores: per recording, one per candidate."""
    present = detected_gru = detected_fft = found = 0
    false_gru = false_fft = 0
    minutes = 0.0
    by_class = defaultdict(lambda: [0, 0, 0])               # present, found, detected by the GRU
    matched_scores, unmatched_scores = [], []
    for r, s in zip(results, scores, strict=True):
        c = r["cands"]
        window, prom, matched = c[:, 0].astype(int), c[:, 6], c[:, 8].astype(int)
        minutes += r["minutes"]
        false_gru += int(((matched < 0) & (s > threshold)).sum())
        false_fft += int(((matched < 0) & (prom > fft_threshold)).sum())
        matched_scores.append(s[matched >= 0]); unmatched_scores.append(s[matched < 0])
        for w, drone, cls, *_ in r["gt"]:                # run_sequence keeps only present drones here
            hits = (window == w) & (matched == drone)
            present += 1; by_class[cls][0] += 1
            if hits.any():
                found += 1; by_class[cls][1] += 1
                if (s[hits] > threshold).any():
                    detected_gru += 1; by_class[cls][2] += 1
                if (prom[hits] > fft_threshold).any():
                    detected_fft += 1
    pos, neg = np.concatenate(matched_scores), np.concatenate(unmatched_scores)
    return {"present": present, "found": found / present, "detected_gru": detected_gru / present,
            "detected_fft": detected_fft / present, "false_per_min_gru": false_gru / minutes,
            "false_per_min_fft": false_fft / minutes, "box_auc_gru": float(phase1.auc(pos, neg)),
            "by_class": {cls: {"present": v[0], "found": v[1] / v[0], "detected_gru": v[2] / v[0]}
                         for cls, v in sorted(by_class.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(8)
    prm = json.loads((phase1.OUT_DIR / "phase1_find_params.json").read_text())
    prm["max_cands"] = MAX_CANDS
    _, val_seq = load_splits("phase3")
    models = {arm: scored_models(arm) for arm in ARMS}
    results, cand_packets, cand_counts = [], [], []
    for s in val_seq:                                        # one recording in memory at a time
        r, packets, log_count = candidates(s, prm)
        del r["win_starts"]
        results.append(r); cand_packets.append(packets); cand_counts.append(log_count)
        print(f"{s}: {len(packets)} candidate boxes, {len(r['gt'])} present drone-windows", flush=True)
    report = {"claim": __doc__.split("Claim")[1].strip(), "max_cands": MAX_CANDS, "params": prm, "arms": {}}
    for arm, (feature_set, constant_gap, stats, seeds) in models.items():
        per_seed = []
        for model, threshold in seeds:
            scores = [predict(model, tensors(pk, lc, feature_set, constant_gap, stats))
                      if len(pk) else np.zeros(0) for pk, lc in zip(cand_packets, cand_counts)]
            per_seed.append(summarise(results, scores, threshold, prm["fft_threshold"]))
        mean = {k: float(np.mean([r[k] for r in per_seed])) for k, v in per_seed[0].items() if not isinstance(v, dict)}
        report["arms"][arm] = {"mean": mean, "seeds": per_seed}
        print(f"{arm}: found {mean['found']:.3f}  detected GRU {mean['detected_gru']:.3f} / FFT {mean['detected_fft']:.3f}"
              f"  false/min GRU {mean['false_per_min_gru']:.2f} / FFT {mean['false_per_min_fft']:.2f}"
              f"  box AUC {mean['box_auc_gru']:.3f}", flush=True)
        for cls, v in per_seed[0]["by_class"].items():
            print(f"    seed 0 {cls:22} present {v['present']:5}  found {v['found']:.3f}  detected {v['detected_gru']:.3f}")
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
