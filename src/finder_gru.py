"""Score the GRU on boxes the Phase 1 finder proposes, not FRED's boxes.

python -m src.finder_gru --out FILE [--arms time_gru time_gru_aug]

Fixed before any result: the finder's frozen parameters
(experiments/phase1_find_params.json, one box per 0.5 s window), the Phase 3
validation recordings, and the Phase 3 GRUs at rate 0.01 with their own alarm
thresholds. A proposal is a drone hit when its IoU with a drone's swept box is
above 0.1, as in Phase 1. Events inside each proposal become packets exactly as
for FRED's boxes; the event count is standardised with Phase 3 training statistics.

Reported per scorer (the GRU's 5 seeds, and the frozen FFT threshold 3.25):
- proposal AUC: hit proposals against missed ones
- detection recall: share of windows with a drone present (12 or more of ~15
  annotation frames) where a hitting proposal passes the threshold
- false detections per minute of camera time: passing proposals that hit no drone
- the same two after a 3-of-4 vote over consecutive windows of the recording
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import phase1_fft as phase1
from src import phase1_find as finder
from src.data.cnn_windows import load_splits
from src.data.event_packets import FEATURES, load_packets, packets_from_events
from src.models.packet_models import PACKET_ARMS
from src.train.packets import predict, tensors

RUNS = Path("runs/phase3-2026-11-24/lr0.01")
FFT_THRESHOLD = 3.25


def proposals(name, prm):
    """Finder output for one recording, plus packets for each proposal."""
    ev = np.load(finder.seq_dir(name) / "Event/events.npy", mmap_mode="r")
    x = np.asarray(ev["x"]).astype(np.int32)
    y = np.asarray(ev["y"]).astype(np.int32)
    t = np.asarray(ev["t"]).astype(np.int64)
    p = np.asarray(ev["p"]).astype(np.float32)
    res = finder.run_sequence(name, prm, ev=(x, y, t), tracks=finder.load_tracks(name))
    packets, counts = [], []
    for row in res["cands"]:
        w, x1, y1, x2, y2 = int(row[0]), *row[1:5]
        w0 = int(res["win_starts"][w])
        i0, i1 = np.searchsorted(t, [w0, w0 + finder.WIN_US])
        xs, ys = x[i0:i1], y[i0:i1]
        inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
        tt = (t[i0:i1][inside] - w0) / 1000.0
        packets.append(packets_from_events(tt, p[i0:i1][inside], ((xs[inside] - x1) / max(x2 - x1, 1)).astype(np.float32),
                                           ((ys[inside] - y1) / max(y2 - y1, 1)).astype(np.float32)))
        counts.append(int(inside.sum()))
    return res, packets, np.log1p(np.array(counts, dtype=np.float32))


def vote(windows, alarm, k=3, n=4):
    """Confirmed per proposal: at least k alarms among the last n windows (any proposal)."""
    fired = {}
    for w, a in zip(windows, alarm):
        fired[w] = fired.get(w, False) or bool(a)
    return np.array([sum(fired.get(w - j, False) for j in range(n)) >= k and a for w, a in zip(windows, alarm)])


def summarise(results, scores, threshold):
    """Detection recall, false detections per camera-minute, and the same after voting."""
    det = det_v = gt_total = false = false_v = 0
    minutes = 0.0
    for res, s in zip(results, scores):
        c = res["cands"]
        alarm = s > threshold
        confirmed = vote(c[:, 0].astype(int), alarm)
        hit = c[:, 8] >= 0
        minutes += res["minutes"]
        false += int((alarm & ~hit).sum())
        false_v += int((confirmed & ~hit).sum())
        by_window = {}
        for (w, *_), a, v, h, d in zip(c, alarm, confirmed, hit, c[:, 8]):
            if h:
                by_window.setdefault((int(w), int(d)), [False, False])
                by_window[(int(w), int(d))][0] |= bool(a)
                by_window[(int(w), int(d))][1] |= bool(v)
        for (w, i, *_rest) in res["gt"]:
            gt_total += 1
            a, v = by_window.get((int(w), int(i)), (False, False))
            det += a
            det_v += v
    return {"recall": det / gt_total, "false_per_min": false / minutes,
            "recall_voted": det_v / gt_total, "false_per_min_voted": false_v / minutes,
            "drone_windows": gt_total, "camera_minutes": minutes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["time_gru", "time_gru_aug"])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    prm = json.loads((phase1.OUT_DIR / "phase1_find_params.json").read_text())
    train_seq, val_seq = load_splits("phase3")
    _, train_count, _, _ = load_packets(train_seq)
    stats = (train_count.mean(), train_count.std())
    results, packets, counts = [], [], []
    for name in val_seq:
        res, pk, ct = proposals(name, prm)
        results.append(res); packets.append(pk); counts.append(ct)
        hits = int((res["cands"][:, 8] >= 0).sum())
        print(f"{name}: {len(pk)} proposals, {hits} on a drone, {len(res['gt'])} drone windows", flush=True)
    sizes = [len(p) for p in packets]
    flat_packets = [q for pk in packets for q in pk]
    flat_counts = np.concatenate(counts)
    hit = np.concatenate([r["cands"][:, 8] >= 0 for r in results])
    split = lambda v: np.split(v, np.cumsum(sizes)[:-1])
    report = {"recordings": val_seq, "proposals": int(len(hit)), "hits": int(hit.sum()), "scorers": {}}
    prom = np.concatenate([r["cands"][:, 6] for r in results])
    report["scorers"]["FFT"] = {"auc": float(phase1.auc(prom[hit], prom[~hit])), **summarise(results, split(prom), FFT_THRESHOLD)}
    report["scorers"]["finder only"] = summarise(results, split(np.ones(len(hit))), 0.0)
    for arm in args.arms:
        feature_set, constant_gap, build = PACKET_ARMS[arm.removesuffix("_aug")]
        store = tensors(flat_packets, flat_counts, feature_set, constant_gap, stats)
        meta = json.loads((RUNS / arm / "results.json").read_text())
        seeds = []
        for run in meta["runs"]:
            model = build(len(FEATURES[feature_set]))
            model.load_state_dict(torch.load(RUNS / arm / f"seed_{run['seed']}.pt", map_location="cpu"))
            s = predict(model, store)
            seeds.append({"auc": float(phase1.auc(s[hit], s[~hit])), **summarise(results, split(s), run["threshold"])})
        report["scorers"][arm] = {"seeds": seeds, **{k: float(np.mean([x[k] for x in seeds])) for k in seeds[0]}}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{report['proposals']} proposals, {report['hits']} on a drone")
    for name, r in report["scorers"].items():
        print(f"{name:14} AUC {r.get('auc', float('nan')):.3f}  recall {r['recall']:.3f}  false/min {r['false_per_min']:6.2f}"
              f"  | voted 3 of 4: recall {r['recall_voted']:.3f}  false/min {r['false_per_min_voted']:6.2f}")


if __name__ == "__main__":
    main()
