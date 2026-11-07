"""Phase 1 multi-sequence evaluation of the FFT rotor detector.

What it does, in order:
  1. Runs src/phase1_fft.py on every sequence in experiments/phase1_sequences.json
     (tune, heldout, challenging, development) at full resolution and at the
     simulated 640 px and 320 px sensors (--downsample 2 and 4). Each run writes
     experiments/csv/<split>_<seq>[_ds2|_ds4].csv. Existing CSVs are reused
     unless --force is given.
  2. Picks ONE threshold for the FFT score and ONE for the count-only baseline
     on the "tune" sequences only (full resolution), with this rule:
         lowest threshold whose false-alarm rate on ACTIVE background windows,
         pooled over all tune sequences, is <= 2 %.
     Why this rule: it fixes the false-alarm budget on the hardest negatives
     we have, and it does not depend on how many background boxes we chose
     per drone (precision and F1 do: 6 background boxes per drone is our
     choice, not a property of the world). The thresholds are frozen in
     experiments/phase1_threshold.json.
  3. Reports precision / recall / AUC / false alarms on heldout and
     challenging, per drone type, FFT vs count-only, against shifted and
     active backgrounds. train/36 is reported separately as "development"
     (it was used to design the detector). Challenging is never tuned on.
  4. Detection rate and AUC vs box width (all sequences, full resolution).
  5. Recall and AUC per drone type at 1280 / 640 / 320 px sensor width.
Outputs: experiments/phase1_multiseq_results.txt, phase1_threshold.json,
         phase1_width_bins.png, phase1_downsample.png

Run from the repo root:
    python src/phase1_multiseq.py            # run what is missing, then report
    python src/phase1_multiseq.py --force    # re-run every sequence
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_fft as p1  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SETS_FILE = p1.OUT_DIR / "phase1_sequences.json"
RESULTS_FILE = p1.OUT_DIR / "phase1_multiseq_results.txt"
FA_TARGET = 0.02                         # tuning rule: <= 2 % false alarms on active backgrounds
DOWNSAMPLES = (1, 2, 4)
WIDTH_EDGES = [0, 20, 30, 45, 70, 100, 150, 10_000]   # box width bins, px (full sensor)


# ----------------------------------------------------------------------------
# 1. Run the detector on every sequence
# ----------------------------------------------------------------------------
def _run_one(job):
    """Worker for the process pool: one (sequence, downsample) run."""
    seq, ds = job
    p1.run_sequence(seq, ds)
    return job


def run_all(seqs, force, jobs):
    todo, missing = [], []
    for seq in seqs:
        if not p1.events_path(seq).exists():
            missing.append(seq); continue
        for ds in DOWNSAMPLES:
            if force or not p1.csv_path(seq, ds).exists():
                todo.append((seq, ds))
    if missing:
        print("MISSING data, skipped:", missing, file=sys.stderr)
    # biggest jobs first would be better; keep the order simple
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for seq, ds in pool.map(_run_one, todo):
            print(f"finished {seq} ds{ds}", file=sys.stderr)
    return missing


# ----------------------------------------------------------------------------
# 2. Load the CSVs
# ----------------------------------------------------------------------------
def load_rows(seqs, ds):
    """All CSV rows of these sequences as a dict of numpy arrays."""
    cols = {k: [] for k in ("sequence", "track_id", "is_background", "score", "event_count",
                            "drone_class", "background_type", "box_width_px")}
    for seq in seqs:
        path = p1.csv_path(seq, ds)
        if not path.exists():
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                for k in cols:
                    cols[k].append(r[k])
    out = {k: np.array(v) for k, v in cols.items()}
    out["score"] = out["score"].astype(float)
    out["event_count"] = out["event_count"].astype(float)
    out["box_width_px"] = out["box_width_px"].astype(float)
    out["is_background"] = out["is_background"].astype(int)
    return out


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


# ----------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------
def tune_threshold(neg_scores):
    """Lowest threshold t with mean(neg > t) <= FA_TARGET."""
    neg = np.sort(np.asarray(neg_scores, float))
    k = int(np.ceil((1 - FA_TARGET) * len(neg))) - 1      # index of the 98th-percentile value
    return float(neg[max(k, 0)])


def metrics(pos, neg, thr):
    """Precision, recall, AUC, false alarms per window and per minute."""
    tp = int((pos > thr).sum()); fp = int((neg > thr).sum())
    bg_minutes = len(neg) * p1.HOP_S / 60      # each window adds HOP_S of new box-time
    return dict(n_pos=len(pos), n_neg=len(neg),
                auc=p1.auc(pos, neg),
                precision=tp / (tp + fp) if tp + fp else float("nan"),
                recall=tp / len(pos) if len(pos) else float("nan"),
                fa_rate=fp / len(neg) if len(neg) else float("nan"),
                fa_per_min=fp / bg_minutes if bg_minutes else float("nan"))


def fmt(x, nd=3):
    return "  nan" if x != x else f"{x:.{nd}f}"


def table(d, thr, title, lines):
    """Per drone type x background type x scorer table."""
    lines.append(f"\n{title}")
    lines.append(f"{'drone type':<22}{'bg':<8}{'scorer':<7}{'n_pos':>6}{'n_neg':>7}{'AUC':>7}"
                 f"{'prec':>7}{'recall':>7}{'FA/win':>8}{'FA/min':>8}")
    types = sorted(set(d["drone_class"])) + ["ALL TYPES"]
    for ty in types:
        m_ty = np.ones(len(d["score"]), bool) if ty == "ALL TYPES" else d["drone_class"] == ty
        pos_m = m_ty & (d["is_background"] == 0)
        for bg in ("shifted", "active"):
            neg_m = m_ty & (d["background_type"] == bg)
            for scorer, col, t in (("FFT", "score", thr["fft_threshold"]),
                                   ("count", "event_count", thr["count_threshold"])):
                r = metrics(d[col][pos_m], d[col][neg_m], t)
                lines.append(f"{ty:<22}{bg:<8}{scorer:<7}{r['n_pos']:>6}{r['n_neg']:>7}{fmt(r['auc']):>7}"
                             f"{fmt(r['precision']):>7}{fmt(r['recall']):>7}{fmt(r['fa_rate']):>8}"
                             f"{fmt(r['fa_per_min'], 2):>8}")


def per_sequence_table(d, lines):
    """How well active backgrounds match the drone's event counts, per sequence."""
    lines.append(f"{'sequence':<11}{'track':<6}{'drone type':<22}{'n_drone':>8}{'drone cnt':>10}"
                 f"{'active cnt':>11}{'shift cnt':>10}{'act/drone':>10}"
                 f"{'AUCfft sh':>10}{'AUCfft act':>11}{'AUCcnt act':>11}")
    for seq in dict.fromkeys(d["sequence"]):
        for tid in dict.fromkeys(d["track_id"][d["sequence"] == seq]):
            m = (d["sequence"] == seq) & (d["track_id"] == tid)
            dr = m & (d["is_background"] == 0)
            ac = m & (d["background_type"] == "active")
            sh = m & (d["background_type"] == "shifted")
            med = lambda mm: np.median(d["event_count"][mm]) if mm.any() else float("nan")
            lines.append(f"{seq:<11}{tid:<6}{d['drone_class'][dr][0] if dr.any() else '?':<22}{dr.sum():>8}"
                         f"{med(dr):>10.0f}{med(ac):>11.0f}{med(sh):>10.0f}{fmt(med(ac) / med(dr), 3):>10}"
                         f"{fmt(p1.auc(d['score'][dr], d['score'][sh])):>10}"
                         f"{fmt(p1.auc(d['score'][dr], d['score'][ac])):>11}"
                         f"{fmt(p1.auc(d['event_count'][dr], d['event_count'][ac])):>11}")


# ----------------------------------------------------------------------------
# 4. Box width analysis
# ----------------------------------------------------------------------------
def width_analysis(d, thr, lines):
    edges = WIDTH_EDGES
    labels = [f"{a}-{b}" if b < 10_000 else f">{a}" for a, b in zip(edges[:-1], edges[1:])]
    w_bin = np.digitize(d["box_width_px"], edges[1:-1])      # 0 .. len(labels)-1
    res = {k: [] for k in ("n", "rec_fft", "rec_cnt", "auc_fft_sh", "auc_fft_act", "auc_cnt_act")}
    per_type = {}                                            # drone type -> [(width bin, row)]
    lines.append(f"{'width px':<10}{'drone type':<22}{'n_pos':>6}{'n_act':>6}{'rec FFT':>8}{'rec cnt':>8}"
                 f"{'AUCfft sh':>10}{'AUCfft act':>11}{'AUCcnt act':>11}")
    types = sorted(set(d["drone_class"])) + ["ALL TYPES"]
    for b, lab in enumerate(labels):
        for ty in types:
            m_ty = np.ones(len(w_bin), bool) if ty == "ALL TYPES" else d["drone_class"] == ty
            pos = m_ty & (w_bin == b) & (d["is_background"] == 0)
            sh = m_ty & (w_bin == b) & (d["background_type"] == "shifted")
            ac = m_ty & (w_bin == b) & (d["background_type"] == "active")
            if pos.sum() == 0:
                continue
            row = dict(n=int(pos.sum()),
                       rec_fft=np.mean(d["score"][pos] > thr["fft_threshold"]),
                       rec_cnt=np.mean(d["event_count"][pos] > thr["count_threshold"]),
                       auc_fft_sh=p1.auc(d["score"][pos], d["score"][sh]),
                       auc_fft_act=p1.auc(d["score"][pos], d["score"][ac]),
                       auc_cnt_act=p1.auc(d["event_count"][pos], d["event_count"][ac]))
            lines.append(f"{lab:<10}{ty:<22}{row['n']:>6}{ac.sum():>6}{fmt(row['rec_fft']):>8}"
                         f"{fmt(row['rec_cnt']):>8}{fmt(row['auc_fft_sh']):>10}{fmt(row['auc_fft_act']):>11}"
                         f"{fmt(row['auc_cnt_act']):>11}")
            if ty == "ALL TYPES":
                for k in res:
                    res[k].append(row[k])
            else:
                per_type.setdefault(ty, []).append((b, row))
    # plot: all types pooled, plus FFT recall per type
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    x = np.arange(len(labels))
    all_b = [b for b in range(len(labels)) if (np.digitize(d["box_width_px"][d["is_background"] == 0],
                                                           edges[1:-1]) == b).any()]
    ax[0].plot(all_b, res["rec_fft"], "o-", label="FFT recall")
    ax[0].plot(all_b, res["rec_cnt"], "s--", label="count recall")
    ax[1].plot(all_b, res["auc_fft_sh"], "o-", label="FFT vs shifted")
    ax[1].plot(all_b, res["auc_fft_act"], "o-", label="FFT vs active")
    ax[1].plot(all_b, res["auc_cnt_act"], "s--", label="count vs active")
    ax[1].axhline(0.5, color="k", lw=0.6, ls=":")
    for ty, val in per_type.items():
        ax[2].plot([b for b, _ in val], [r["auc_fft_act"] for _, r in val], "o-", label=ty)
    ax[2].axhline(0.5, color="k", lw=0.6, ls=":")
    titles = ["recall at frozen threshold (all types)", "AUC (all types)", "FFT AUC vs active, per drone type"]
    for a, t in zip(ax, titles):
        a.set_xticks(x); a.set_xticklabels(labels, rotation=30); a.set_xlabel("box width (px, 1280-wide sensor)")
        a.set_title(t, fontsize=10); a.legend(fontsize=8); a.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(p1.OUT_DIR / "phase1_width_bins.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Downsampling analysis
# ----------------------------------------------------------------------------
def downsample_analysis(seqs, thr, lines):
    data = {ds: load_rows(seqs, ds) for ds in DOWNSAMPLES}
    types = sorted(set(data[1]["drone_class"])) + ["ALL TYPES"]
    lines.append(f"{'drone type':<22}{'sensor':>7}{'n_pos':>6}{'rec FFT':>8}{'FA act':>7}{'AUCfft sh':>10}"
                 f"{'AUCfft act':>11}{'AUCcnt act':>11}")
    plot = {}
    for ty in types:
        for ds in DOWNSAMPLES:
            d = data[ds]
            m_ty = np.ones(len(d["score"]), bool) if ty == "ALL TYPES" else d["drone_class"] == ty
            pos = m_ty & (d["is_background"] == 0)
            sh = m_ty & (d["background_type"] == "shifted")
            ac = m_ty & (d["background_type"] == "active")
            if not pos.any():
                continue
            a_act = p1.auc(d["score"][pos], d["score"][ac])
            lines.append(f"{ty:<22}{p1.SENSOR_W // ds:>7}{pos.sum():>6}"
                         f"{fmt(np.mean(d['score'][pos] > thr['fft_threshold'])):>8}"
                         f"{fmt(np.mean(d['score'][ac] > thr['fft_threshold'])):>7}"
                         f"{fmt(p1.auc(d['score'][pos], d['score'][sh])):>10}{fmt(a_act):>11}"
                         f"{fmt(p1.auc(d['event_count'][pos], d['event_count'][ac])):>11}")
            plot.setdefault(ty, []).append((p1.SENSOR_W // ds, a_act,
                                            np.mean(d["score"][pos] > thr["fft_threshold"])))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for ty, v in plot.items():
        w = [a for a, _, _ in v]
        ax[0].plot(w, [b for _, b, _ in v], "o-", label=ty, lw=2.5 if ty == "ALL TYPES" else 1)
        ax[1].plot(w, [c for _, _, c in v], "o-", label=ty, lw=2.5 if ty == "ALL TYPES" else 1)
    for a, t in zip(ax, ("FFT AUC vs active backgrounds", "FFT recall at frozen threshold")):
        a.set_xscale("log", base=2); a.set_xticks([320, 640, 1280]); a.set_xticklabels(["320", "640", "1280"])
        a.set_xlabel("simulated sensor width (px)"); a.set_title(t, fontsize=10); a.legend(fontsize=7)
        a.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(p1.OUT_DIR / "phase1_downsample.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-run sequences even if their CSV exists")
    ap.add_argument("--jobs", type=int, default=4, help="sequences run in parallel (default 4)")
    a = ap.parse_args()

    sets = json.loads(SETS_FILE.read_text())
    groups = {k: sets[k] for k in ("tune", "heldout", "challenging", "development")}
    all_seqs = [s for g in groups.values() for s in g]
    missing = run_all(all_seqs, a.force, a.jobs)

    L = []
    L.append("Phase 1 multi-sequence evaluation: FFT rotor detector vs count-only baseline")
    L.append(f"Locked settings: {p1.BIN_US / 1000:g} ms bins, {p1.F_LO:g}-{p1.F_HI:g} Hz, "
             f"{p1.WIN_S} s windows, {p1.HOP_S} s step. Written by src/phase1_multiseq.py.")
    if missing:
        L.append(f"MISSING (no events.npy, skipped): {missing}")
    for g, s in groups.items():
        L.append(f"{g:<12}: {', '.join(s)}")

    # ---- tune --------------------------------------------------------------
    tune = load_rows(groups["tune"], 1)
    act = tune["background_type"] == "active"
    thr = dict(fft_threshold=tune_threshold(tune["score"][act]),
               count_threshold=tune_threshold(tune["event_count"][act]),
               rule=f"lowest threshold with false-alarm rate <= {FA_TARGET:.0%} on active background "
                    f"windows pooled over the tune sequences (decision: score > threshold)",
               tuned_on=groups["tune"], n_active_tune_windows=int(act.sum()),
               settings=dict(bin_ms=p1.BIN_US / 1000, f_lo_hz=p1.F_LO, f_hi_hz=p1.F_HI,
                             window_s=p1.WIN_S, step_s=p1.HOP_S))
    p1.THRESHOLD_FILE.write_text(json.dumps(thr, indent=2) + "\n")
    L.append(f"\nFROZEN THRESHOLDS ({p1.THRESHOLD_FILE.name}): FFT prominence > {thr['fft_threshold']:.4f}, "
             f"event_count > {thr['count_threshold']:.0f}")
    L.append(f"Rule: {thr['rule']}. {thr['n_active_tune_windows']} active tune windows.")
    L.append("Why: fixes the false-alarm budget on the hardest negatives, and does not depend on the")
    L.append("number of background boxes per drone (precision and F1 do). One global count threshold")
    L.append("is crude because event counts scale with box size and distance; that is the baseline's problem.")
    L.append("\nNotes on the columns:")
    L.append("  AUC = P[drone window scores higher than background window], threshold-free.")
    L.append("  prec/recall/FA use the frozen thresholds. Precision depends on our ratio of background to")
    L.append("  drone windows (up to 3 shifted + 3 active boxes per drone), so compare it between rows, not")
    L.append("  to the outside world. FA/min = false alarms per minute of background box-time")
    L.append(f"  (each window adds {p1.HOP_S} s; per single background box).")

    # ---- per split tables --------------------------------------------------
    for g, title in (("heldout", "HELDOUT (frozen threshold, never tuned on)"),
                     ("challenging", "CHALLENGING (frozen threshold, never tuned on)"),
                     ("development", "DEVELOPMENT train/36 (detector designed on it earlier; not a test)"),
                     ("tune", "TUNE (threshold chosen here; in-sample, for reference only)")):
        d = load_rows(groups[g], 1)
        table(d, thr, f"=== {title} ===", L)

    # ---- count matching ----------------------------------------------------
    L.append("\n=== ACTIVE BACKGROUNDS: event counts vs the drone box (per sequence, per track) ===")
    L.append("cnt = median events per 0.5 s window. act/drone = active median / drone median.")
    everything = load_rows(all_seqs, 1)
    per_sequence_table(everything, L)

    # ---- width -------------------------------------------------------------
    L.append("\n=== BOX WIDTH (all sequences incl. tune and development, full resolution) ===")
    L.append("Negatives in a width bin are the backgrounds of drones in that bin (same box size).")
    width_analysis(everything, thr, L)

    # ---- downsampling ------------------------------------------------------
    L.append("\n=== SIMULATED SMALLER SENSOR (heldout + challenging pooled, frozen threshold) ===")
    L.append("x, y and boxes integer-divided by 2 / 4; events on the same coarse pixel kept separate.")
    downsample_analysis(groups["heldout"] + groups["challenging"], thr, L)

    RESULTS_FILE.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
