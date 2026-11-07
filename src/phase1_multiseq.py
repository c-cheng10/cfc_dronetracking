"""Phase 1 multi-sequence evaluation of the FFT rotor detector.

What it does, in order:
  1. Runs src/phase1_fft.py on every sequence in experiments/phase1_sequences.json
     (tune, heldout, challenging, development) at full resolution and at the
     simulated 640 px and 320 px sensors (--downsample 2 and 4). Each run writes
     experiments/csv/<split>_<seq>[_ds2|_ds4].csv. Existing CSVs are reused
     unless --force is given.
  2. On the "tune" sequences ONLY (full resolution) it chooses and freezes:
     a. The peak-stability rule that rejects flicker (lights hold one exact
        frequency for seconds, rotors drift). See STABILITY TUNING below.
     b. Three logistic-regression models (plain NumPy, Newton's method):
          basic      log(event_count+1), log(prominence)
          stab       + stability
          stab_area  + log(box area)
        The one with the best leave-one-tune-sequence-out recall is named
        "combined_model": the number the CfC has to beat.
     c. One threshold per scorer, all with the same rule:
            lowest threshold whose false-alarm rate on ACTIVE background
            windows, pooled over the tune sequences, is <= 2 %.
        Why: it fixes the false-alarm budget on the hardest negatives, and it
        does not depend on how many background boxes we chose per drone
        (precision and F1 do).
     Everything is saved in experiments/phase1_threshold.json.
  3. Regenerates the full-resolution CSVs so they carry the derived columns
     (stability, score_fft_stab, score_lr_*).
  4. Reports precision / recall / AUC / false alarms per minute on heldout and
     challenging, per drone type, for every scorer, against shifted and active
     backgrounds. train/36 is reported separately as "development". The
     challenging split is never used for any choice.
  5. Drone windows lost to the stability rule, flicker check, count matching,
     box width, simulated smaller sensors, and generated notes.
Outputs: experiments/phase1_multiseq_results.txt, phase1_threshold.json,
         phase1_width_bins.png, phase1_downsample.png

STABILITY TUNING. Feature: see peak_stability() in phase1_fft.py (fraction of
the same box's windows 0.5..N s away whose peak is within +-1 FFT bin). We try
N in STAB_N_GRID and rejection cut s_cut in STAB_CUT_GRID; a window is
rejected when stability > s_cut. For each pair the FFT threshold is re-tuned
with the 2 % rule, then we keep the pair with the highest tune-set drone
recall, among pairs that reject at most MAX_DRONE_REJECT of tune drone
windows (ties: fewer rejected drone windows). Caveat: the tune set contains
almost no flickering lights, so this mostly tunes the COST of the rule (lost
drone windows); its benefit is only measured on heldout / challenging.

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
FA_TARGET = 0.02                         # threshold rule: <= 2 % false alarms on active backgrounds
DOWNSAMPLES = (1, 2, 4)
WIDTH_EDGES = [0, 20, 30, 45, 70, 100, 150, 10_000]   # box width bins, px (full sensor)
STAB_N_GRID = (1.0, 2.0, 3.0, 4.0)       # seconds each side
STAB_CUT_GRID = tuple(np.round(np.arange(0.30, 0.96, 0.05), 2))
MAX_DRONE_REJECT = 0.05                  # stability rule may drop at most 5 % of tune drone windows
L2 = 1e-3                                # ridge penalty for the logistic models
LR_VARIANTS = {"basic": ["log_count", "log_prom"],
               "stab": ["log_count", "log_prom", "stability"],
               "stab_area": ["log_count", "log_prom", "stability", "log_area"]}
# scorer label -> column of the loaded data
SCORERS = {"FFT": "score", "FFT+stab": "score_fft_stab", "count": "event_count",
           "LR basic": "score_lr_basic", "LR stab": "score_lr_stab", "LR stab+area": "score_lr_stab_area"}


# ----------------------------------------------------------------------------
# 1. Run the detector on every sequence
# ----------------------------------------------------------------------------
def _run_one(job):
    """Worker for the process pool: one (sequence, downsample) run."""
    seq, ds = job
    p1.run_sequence(seq, ds)
    return job


def run_jobs(todo, jobs):
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for seq, ds in pool.map(_run_one, todo):
            print(f"finished {seq} ds{ds}", file=sys.stderr)


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
    run_jobs(todo, jobs)
    return missing


# ----------------------------------------------------------------------------
# 2. Load the CSVs (base columns only; derived scores are recomputed here)
# ----------------------------------------------------------------------------
NUMERIC = ("score", "event_count", "box_width_px", "box_height_px", "box_x1", "box_y1",
           "t_start_us", "peak_hz")


def load_rows(seqs, ds):
    """All CSV rows of these sequences as a dict of numpy arrays."""
    names = ("sequence", "track_id", "is_background", "drone_class", "background_type",
             "box_name") + NUMERIC
    cols = {k: [] for k in names}
    for seq in seqs:
        path = p1.csv_path(seq, ds)
        if not path.exists():
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                for k in cols:
                    cols[k].append(r[k])
    out = {k: np.array(v) for k, v in cols.items()}
    for k in NUMERIC:
        out[k] = out[k].astype(float)
    out["is_background"] = out["is_background"].astype(int)
    out["key"] = np.array([p1.stability_key(*a) for a in zip(
        out["sequence"], out["track_id"], out["background_type"], out["box_name"],
        out["box_x1"], out["box_y1"])])
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


def type_masks(d):
    """[(drone type, mask)], every type plus ALL TYPES."""
    out = [(ty, d["drone_class"] == ty) for ty in sorted(set(d["drone_class"]))]
    return out + [("ALL TYPES", np.ones(len(d["score"]), bool))]


def split_metrics(d, thr):
    """{(type, bg, scorer): metrics} for one split."""
    res = {}
    for ty, m_ty in type_masks(d):
        pos_m = m_ty & (d["is_background"] == 0)
        for bg in ("shifted", "active"):
            neg_m = m_ty & (d["background_type"] == bg)
            for label, col in SCORERS.items():
                res[(ty, bg, label)] = metrics(d[col][pos_m], d[col][neg_m], thr[col])
    return res


def table(res, title, lines):
    lines.append(f"\n{title}")
    lines.append(f"{'drone type':<22}{'bg':<8}{'scorer':<13}{'n_pos':>6}{'n_neg':>7}{'AUC':>7}"
                 f"{'prec':>7}{'recall':>7}{'FA/win':>8}{'FA/min':>8}")
    last_ty = None
    for (ty, bg, label), r in res.items():
        if last_ty is not None and ty != last_ty:
            lines.append("")
        last_ty = ty
        lines.append(f"{ty:<22}{bg:<8}{label:<13}{r['n_pos']:>6}{r['n_neg']:>7}{fmt(r['auc']):>7}"
                     f"{fmt(r['precision']):>7}{fmt(r['recall']):>7}{fmt(r['fa_rate']):>8}"
                     f"{fmt(r['fa_per_min'], 2):>8}")


# ----------------------------------------------------------------------------
# 4. Stability rule and logistic models (tune set only)
# ----------------------------------------------------------------------------
def tune_stability(d, tune_m, lines):
    """Grid over (N, s_cut); returns (n_s, s_cut, stability array for chosen N)."""
    pos = tune_m & (d["is_background"] == 0)
    act = tune_m & (d["background_type"] == "active")
    lines.append(f"{'N s':>5}{'s_cut':>7}{'thr':>8}{'recall':>8}{'drone rej':>10}{'act rej':>9}")
    stabs, best = {}, None
    base_thr = tune_threshold(d["score"][act])
    lines.append(f"{'none':>5}{'':>7}{base_thr:>8.3f}{np.mean(d['score'][pos] > base_thr):>8.3f}"
                 f"{0:>10.3f}{0:>9.3f}")
    for n_s in STAB_N_GRID:
        stab = p1.peak_stability(d["key"], d["t_start_us"], d["peak_hz"], n_s)
        stabs[n_s] = stab
        for cut in STAB_CUT_GRID:
            rej = stab > cut                                    # NaN > cut is False: kept
            sc = np.where(rej, 0.0, d["score"])
            thr = tune_threshold(sc[act])
            rec = np.mean(sc[pos] > thr)
            drej = rej[pos].mean()
            lines.append(f"{n_s:>5g}{cut:>7.2f}{thr:>8.3f}{rec:>8.3f}{drej:>10.3f}{rej[act].mean():>9.3f}")
            if drej <= MAX_DRONE_REJECT:
                cand = (rec, -drej, -n_s, cut)
                if best is None or cand > best[0]:
                    best = (cand, n_s, cut)
    _, n_s, cut = best
    return n_s, float(cut), stabs[n_s]


def fit_logreg(X, y, l2=L2, iters=30):
    """Logistic regression by Newton's method on standardised features.
    Classes are re-weighted to count equally (there are ~6 background windows
    per drone window). Returns a model dict usable by phase1_fft.lr_score."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Z = np.column_stack([(X - mean) / std, np.ones(len(X))])   # last column = bias
    n, n_pos = len(y), y.sum()
    sw = np.where(y == 1, n / (2 * n_pos), n / (2 * (n - n_pos)))
    w = np.zeros(Z.shape[1])
    pen = np.full(Z.shape[1], l2); pen[-1] = 0.0                # no penalty on the bias
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (sw * (p - y)) / n + pen * w
        hess = (Z * (sw * p * (1 - p))[:, None]).T @ Z / n + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return dict(mean=mean.tolist(), std=std.tolist(), weights=w[:-1].tolist(), bias=float(w[-1]))


def lr_X(d, names):
    return p1.lr_features(names, d["event_count"], d["score"], d["stability"],
                          d["box_width_px"], d["box_height_px"])


def fit_models(d, tune_seqs, lines):
    """Fit every variant on all tune windows; pick the combined model by
    leave-one-tune-sequence-out recall at the 2 % active-FA threshold."""
    tune_m = np.isin(d["sequence"], tune_seqs)
    y = (d["is_background"] == 0).astype(float)
    models, loso = {}, {}
    for name, feats in LR_VARIANTS.items():
        X = lr_X(d, feats)
        held = np.full(len(y), np.nan)
        for s in tune_seqs:                          # leave one tune sequence out
            tr = tune_m & (d["sequence"] != s)
            te = d["sequence"] == s
            held[te] = p1.lr_score(fit_logreg(X[tr], y[tr]), X[te])
        act = tune_m & (d["background_type"] == "active")
        pos = tune_m & (y == 1)
        thr = tune_threshold(held[act])
        loso[name] = (np.mean(held[pos] > thr), p1.auc(held[pos], held[act]))
        models[name] = dict(features=feats, **fit_logreg(X[tune_m], y[tune_m]))
    lines.append(f"{'model':<11}{'features':<46}{'LOSO recall':>12}{'LOSO AUC act':>13}")
    for name in LR_VARIANTS:
        lines.append(f"{name:<11}{', '.join(LR_VARIANTS[name]):<46}{loso[name][0]:>12.3f}{loso[name][1]:>13.3f}")
    combined = max(loso, key=lambda k: (loso[k][0], loso[k][1]))
    return models, combined, loso


# ----------------------------------------------------------------------------
# 5. Count matching, width, downsampling
# ----------------------------------------------------------------------------
def per_sequence_table(d, lines):
    """How well active backgrounds match the drone's event counts, per sequence.
    Returns [(seq, track, ratio)] for the notes."""
    out = []
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
            ratio = med(ac) / med(dr)
            out.append((seq, tid, ratio, p1.auc(d["score"][dr], d["score"][ac]),
                        p1.auc(d["event_count"][dr], d["event_count"][ac])))
            lines.append(f"{seq:<11}{tid:<6}{d['drone_class'][dr][0] if dr.any() else '?':<22}{dr.sum():>8}"
                         f"{med(dr):>10.0f}{med(ac):>11.0f}{med(sh):>10.0f}{fmt(ratio, 3):>10}"
                         f"{fmt(p1.auc(d['score'][dr], d['score'][sh])):>10}"
                         f"{fmt(out[-1][3]):>11}{fmt(out[-1][4]):>11}")
    return out


def width_analysis(d, thr, lines):
    """Recall and AUC per box-width bin. Returns {bin label: FFT recall} (all types)."""
    edges = WIDTH_EDGES
    labels = [f"{a}-{b}" if b < 10_000 else f">{a}" for a, b in zip(edges[:-1], edges[1:])]
    w_bin = np.digitize(d["box_width_px"], edges[1:-1])      # 0 .. len(labels)-1
    res = {k: [] for k in ("b", "n", "rec_fft", "rec_cnt", "auc_fft_sh", "auc_fft_act", "auc_cnt_act")}
    per_type = {}                                            # drone type -> [(width bin, row)]
    lines.append(f"{'width px':<10}{'drone type':<22}{'n_pos':>6}{'n_act':>6}{'rec FFT':>8}{'rec cnt':>8}"
                 f"{'AUCfft sh':>10}{'AUCfft act':>11}{'AUCcnt act':>11}")
    for b, lab in enumerate(labels):
        for ty, m_ty in type_masks(d):
            pos = m_ty & (w_bin == b) & (d["is_background"] == 0)
            sh = m_ty & (w_bin == b) & (d["background_type"] == "shifted")
            ac = m_ty & (w_bin == b) & (d["background_type"] == "active")
            if pos.sum() == 0:
                continue
            row = dict(b=b, n=int(pos.sum()),
                       rec_fft=np.mean(d["score"][pos] > thr["score"]),
                       rec_cnt=np.mean(d["event_count"][pos] > thr["event_count"]),
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
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(res["b"], res["rec_fft"], "o-", label="FFT recall")
    ax[0].plot(res["b"], res["rec_cnt"], "s--", label="count recall")
    ax[1].plot(res["b"], res["auc_fft_sh"], "o-", label="FFT vs shifted")
    ax[1].plot(res["b"], res["auc_fft_act"], "o-", label="FFT vs active")
    ax[1].plot(res["b"], res["auc_cnt_act"], "s--", label="count vs active")
    ax[1].axhline(0.5, color="k", lw=0.6, ls=":")
    for ty, val in per_type.items():
        ax[2].plot([b for b, _ in val], [r["auc_fft_act"] for _, r in val], "o-", label=ty)
    ax[2].axhline(0.5, color="k", lw=0.6, ls=":")
    titles = ["recall at frozen threshold (all types)", "AUC (all types)", "FFT AUC vs active, per drone type"]
    for a, t in zip(ax, titles):
        a.set_xticks(np.arange(len(labels))); a.set_xticklabels(labels, rotation=30)
        a.set_xlabel("box width (px, 1280-wide sensor)")
        a.set_title(t, fontsize=10); a.legend(fontsize=8); a.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(p1.OUT_DIR / "phase1_width_bins.png", dpi=110)
    plt.close(fig)
    return {labels[b]: (n, r) for b, n, r in zip(res["b"], res["n"], res["rec_fft"])}


def downsample_analysis(seqs, thr, lines):
    """Plain FFT and count per drone type at 1280 / 640 / 320 px. Returns the
    largest |AUC change| vs full resolution, for the notes."""
    data = {ds: load_rows(seqs, ds) for ds in DOWNSAMPLES}
    lines.append(f"{'drone type':<22}{'sensor':>7}{'n_pos':>6}{'rec FFT':>8}{'FA act':>7}{'AUCfft sh':>10}"
                 f"{'AUCfft act':>11}{'AUCcnt act':>11}")
    plot, full, max_delta = {}, {}, 0.0
    for ty, _ in type_masks(data[1]):
        for ds in DOWNSAMPLES:
            d = data[ds]
            m_ty = np.ones(len(d["score"]), bool) if ty == "ALL TYPES" else d["drone_class"] == ty
            pos = m_ty & (d["is_background"] == 0)
            sh = m_ty & (d["background_type"] == "shifted")
            ac = m_ty & (d["background_type"] == "active")
            if not pos.any():
                continue
            aucs = (p1.auc(d["score"][pos], d["score"][sh]), p1.auc(d["score"][pos], d["score"][ac]),
                    p1.auc(d["event_count"][pos], d["event_count"][ac]))
            if ds == 1:
                full[ty] = aucs
            else:
                max_delta = max(max_delta, *(abs(a - b) for a, b in zip(aucs, full[ty])))
            rec = np.mean(d["score"][pos] > thr["score"])
            lines.append(f"{ty:<22}{p1.SENSOR_W // ds:>7}{pos.sum():>6}{fmt(rec):>8}"
                         f"{fmt(np.mean(d['score'][ac] > thr['score'])):>7}"
                         f"{fmt(aucs[0]):>10}{fmt(aucs[1]):>11}{fmt(aucs[2]):>11}")
            plot.setdefault(ty, []).append((p1.SENSOR_W // ds, aucs[1], rec))
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
    return max_delta


# ----------------------------------------------------------------------------
# 6. Stability-rule side effects: lost drone windows, flicker check
# ----------------------------------------------------------------------------
def near_mains_harmonic(f, tol=p1.BIN_HZ):
    """True where f is within tol of a harmonic (>= 2nd) of 50 or 60 Hz mains.
    Used only to REPORT flicker, never to reject anything."""
    f = np.asarray(f, float)
    out = np.zeros(len(f), bool)
    for mains in (50.0, 60.0):
        k = np.round(f / mains)
        out |= (k >= 2) & (np.abs(f - k * mains) <= tol)
    return out


def rejection_table(d, groups, thr, s_cut, lines):
    lines.append(f"{'split':<12}{'drone type':<22}{'n_pos':>6}{'rejected':>9}{'lost hits':>10}")
    lines.append("rejected = drone windows with stability > s_cut; lost hits = drone windows above the plain")
    lines.append("FFT threshold that the rule rejects, as a fraction of those hits.")
    out = {}
    for g in ("tune", "heldout", "challenging", "development"):
        dg = subset(d, np.isin(d["sequence"], groups[g]))
        for ty, m_ty in type_masks(dg):
            pos = m_ty & (dg["is_background"] == 0)
            rej = dg["stability"][pos] > s_cut
            hit = dg["score"][pos] > thr["score"]
            lost = (rej & hit).sum() / max(hit.sum(), 1)
            out[(g, ty)] = (rej.mean(), lost)
            lines.append(f"{g:<12}{ty:<22}{pos.sum():>6}{rej.mean():>9.3f}{lost:>10.3f}")
    return out


def flicker_table(d, thr, combined, lines):
    """Active-box false alarms whose peak sits on a mains harmonic, before and
    after the stability rule (each scorer at its own frozen threshold)."""
    lines.append(f"{'sequence':<11}{'active':>7}{'FA FFT':>8}{'on mains':>9}{'FA FFT+stab':>12}{'on mains':>9}"
                 f"{'FA comb':>8}{'on mains':>9}")
    out = {}
    comb_col = "score_lr_" + combined
    for seq in dict.fromkeys(d["sequence"]):
        ac = (d["sequence"] == seq) & (d["background_type"] == "active")
        mains = near_mains_harmonic(d["peak_hz"][ac])
        fa1 = d["score"][ac] > thr["score"]
        fa2 = d["score_fft_stab"][ac] > thr["score_fft_stab"]
        fa3 = d[comb_col][ac] > thr[comb_col]
        out[seq] = ((fa1 & mains).sum(), (fa2 & mains).sum(), fa1.sum(), fa2.sum())
        lines.append(f"{seq:<11}{ac.sum():>7}{fa1.sum():>8}{(fa1 & mains).sum():>9}{fa2.sum():>12}"
                     f"{(fa2 & mains).sum():>9}{fa3.sum():>8}{(fa3 & mains).sum():>9}")
    return out


# ----------------------------------------------------------------------------
# 7. Generated notes (so a rerun never loses them)
# ----------------------------------------------------------------------------
def write_notes(res, matching, flick, rejections, width_rec, ds_delta, params, groups, d, thr, L):
    L.append("\n=== NOTES (generated from the numbers above) ===")
    comb = "LR " + params["combined_model"].replace("_area", "+area")
    for g in ("heldout", "challenging"):
        r = res[g]
        a = lambda sc, bg="active", ty="ALL TYPES": r[(ty, bg, sc)]["auc"]
        L.append(f"- {g}, ALL TYPES vs active: AUC FFT {a('FFT'):.3f}, FFT+stab {a('FFT+stab'):.3f}, "
                 f"count {a('count'):.3f}, {comb} {a(comb):.3f}. "
                 + ("FFT+stab beats count." if a('FFT+stab') > a('count') else
                    "FFT+stab does NOT beat count, so the rotor signal alone does not do the work."))
        wins = [ty for ty, _ in type_masks(subset(d, np.isin(d["sequence"], groups[g])))
                if ty != "ALL TYPES" and a("FFT+stab", ty=ty) > a("count", ty=ty)]
        L.append(f"  Types where FFT+stab beats count on active: {', '.join(wins) or 'none'}.")
        L.append(f"  Combined vs count on active: {a(comb):.3f} vs {a('count'):.3f}; "
                 f"vs shifted: {a(comb, 'shifted'):.3f} vs {a('count', 'shifted'):.3f}.")
    low = [m for m in matching if m[2] == m[2] and m[2] < 0.5]
    L.append(f"- Count matching: in {len(low)} of {len(matching)} tracks the active boxes have under half the "
             f"drone's median event count; scenes rarely hold anything as busy as a nearby drone, so")
    L.append("  count-only keeps an advantage the active backgrounds were meant to remove.")
    tot_before = sum(v[0] for v in flick.values()); tot_after = sum(v[1] for v in flick.values())
    worst = max(flick, key=lambda s: flick[s][0])
    L.append(f"- Flicker: {tot_before} active-box FFT false alarms peak on a 50/60 Hz mains harmonic (+-2 Hz); "
             f"after the stability rule {tot_after}. Worst sequence {worst}: "
             f"{flick[worst][0]} -> {flick[worst][1]} (all FAs {flick[worst][2]} -> {flick[worst][3]}).")
    edge = (d["background_type"] == "active") & (d["score"] > thr["score"]) & (d["peak_hz"] <= p1.F_LO + 0.1)
    allfa = (d["background_type"] == "active") & (d["score"] > thr["score"])
    L.append(f"- {edge.sum()} of {allfa.sum()} active FFT false alarms peak at the 50 Hz band edge: slow "
             f"(< 50 Hz) changes leaking into the band, not a narrow tone.")
    for g in ("heldout", "challenging"):
        bad = [f"{ty} {v[0]:.1%} (lost hits {v[1]:.1%})" for (gg, ty), v in rejections.items()
               if gg == g and ty != "ALL TYPES"]
        L.append(f"- Drone windows rejected by the stability rule, {g}: {'; '.join(bad)}.")
    L.append(f"- Simulated smaller sensor: the largest AUC change at 640 or 320 px is {ds_delta:.3f}. The "
             f"score sums every event in the box, and integer-dividing")
    L.append("  coordinates only moves box edges by under one coarse pixel, so this simulation cannot answer")
    L.append("  whether a smaller sensor loses the rotor; that needs a per-pixel event-generation model.")
    L.append("- FFT recall at the frozen threshold by box width (all types): "
             + ", ".join(f"{k}: {v[1]:.2f} (n={v[0]})" for k, v in width_rec.items()) + ".")


# ----------------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-run sequences even if their CSV exists")
    ap.add_argument("--jobs", type=int, default=4, help="sequences run in parallel (default 4)")
    ap.add_argument("--no-regen", action="store_true",
                    help="do not rewrite the full-resolution CSVs with the derived columns")
    a = ap.parse_args()

    sets = json.loads(SETS_FILE.read_text())
    groups = {k: sets[k] for k in ("tune", "heldout", "challenging", "development")}
    all_seqs = [s for g in groups.values() for s in g]
    missing = run_all(all_seqs, a.force, a.jobs)

    L = ["Phase 1 multi-sequence evaluation: FFT rotor detector vs count-only and combined baselines",
         f"Locked settings: {p1.BIN_US / 1000:g} ms bins, {p1.F_LO:g}-{p1.F_HI:g} Hz, "
         f"{p1.WIN_S} s windows, {p1.HOP_S} s step. Written by src/phase1_multiseq.py."]
    if missing:
        L.append(f"MISSING (no events.npy, skipped): {missing}")
    for g, s in groups.items():
        L.append(f"{g:<12}: {', '.join(s)}")

    d = load_rows(all_seqs, 1)
    tune_m = np.isin(d["sequence"], groups["tune"])

    # ---- stability rule (tune only) ---------------------------------------
    T = ["\n=== STABILITY TUNING (tune set only) ===",
         "stability = fraction of the same box's windows 0.5..N s away whose peak is within +-1 FFT bin",
         f"({p1.BIN_HZ:g} Hz). Reject when stability > s_cut. FFT threshold re-tuned per row (2 % rule).",
         f"Chosen: highest tune recall with drone rejection <= {MAX_DRONE_REJECT:.0%}. act rej = active "
         "windows rejected."]
    n_s, s_cut, stab = tune_stability(d, tune_m, T)
    d["stability"] = stab
    d["score_fft_stab"] = np.where(stab > s_cut, 0.0, d["score"])

    # ---- logistic models (tune only) --------------------------------------
    M = ["\n=== LOGISTIC MODELS (fit on tune windows: drone vs shifted + active background) ===",
         "Newton's method, standardised features, balanced class weights, L2 = 1e-3.",
         "LOSO = leave one tune sequence out (fit on 4, score the 5th, pool, 2 % rule)."]
    models, combined, loso = fit_models(d, groups["tune"], M)
    for name, model in models.items():
        d[f"score_lr_{name}"] = p1.lr_score(model, lr_X(d, model["features"]))

    # ---- thresholds (tune only) -------------------------------------------
    act = tune_m & (d["background_type"] == "active")
    thr = {col: tune_threshold(d[col][act]) for col in SCORERS.values()}
    for name, model in models.items():
        model["threshold"] = thr[f"score_lr_{name}"]
        model["loso_recall"], model["loso_auc_active"] = loso[name]
    params = dict(
        fft_threshold=thr["score"], count_threshold=thr["event_count"],
        fft_stab_threshold=thr["score_fft_stab"],
        rule=f"lowest threshold with false-alarm rate <= {FA_TARGET:.0%} on active background "
             f"windows pooled over the tune sequences (decision: score > threshold)",
        stability=dict(n_s=n_s, s_cut=s_cut, min_sep_s=0.5, match_hz=p1.BIN_HZ,
                       rule=f"grid N in {list(STAB_N_GRID)} s, s_cut in 0.30..0.95; highest tune recall "
                            f"with <= {MAX_DRONE_REJECT:.0%} of tune drone windows rejected"),
        lr_models=models, combined_model=combined,
        tuned_on=groups["tune"], n_active_tune_windows=int(act.sum()),
        settings=dict(bin_ms=p1.BIN_US / 1000, f_lo_hz=p1.F_LO, f_hi_hz=p1.F_HI,
                      window_s=p1.WIN_S, step_s=p1.HOP_S))
    p1.THRESHOLD_FILE.write_text(json.dumps(params, indent=2) + "\n")

    L.append(f"\nFROZEN ({p1.THRESHOLD_FILE.name}), rule: {params['rule']}; "
             f"{params['n_active_tune_windows']} active tune windows.")
    for label, col in SCORERS.items():
        L.append(f"  {label:<13} score > {thr[col]:.4f}")
    L.append(f"  stability rule: N = {n_s:g} s, reject when stability > {s_cut:.2f}")
    L.append(f"  combined model (the bar for the CfC): {combined}")
    for name, m in models.items():
        L.append(f"  LR {name}: features {m['features']}, weights (standardised) "
                 f"{[round(w, 4) for w in m['weights']]}, bias {m['bias']:.4f}, "
                 f"mean {[round(v, 4) for v in m['mean']]}, std {[round(v, 4) for v in m['std']]}")
    L.append("Why the 2 % rule: fixes the false-alarm budget on the hardest negatives and does not depend")
    L.append("on the number of background boxes per drone (precision and F1 do). A single global count")
    L.append("threshold is crude because counts scale with box size; the LR models can use log box area.")
    L.append("\nColumns: AUC = P[drone window scores higher than background window], threshold-free.")
    L.append("prec/recall/FA use the frozen thresholds. Precision depends on our ratio of background to")
    L.append("drone windows (up to 3 shifted + 3 active boxes per drone): compare between rows only.")
    L.append(f"FA/min = false alarms per minute of single-background-box time (each window adds {p1.HOP_S} s).")
    L += T + M

    # ---- regenerate CSVs with the derived columns -------------------------
    if not a.no_regen:
        run_jobs([(s, 1) for s in all_seqs if s not in missing], a.jobs)

    # ---- per split tables --------------------------------------------------
    res = {}
    for g, title in (("heldout", "HELDOUT (frozen thresholds, never tuned on)"),
                     ("challenging", "CHALLENGING (frozen thresholds, never tuned on)"),
                     ("development", "DEVELOPMENT train/36 (detector designed on it earlier; not a test)"),
                     ("tune", "TUNE (all choices made here; in-sample, for reference only)")):
        res[g] = split_metrics(subset(d, np.isin(d["sequence"], groups[g])), thr)
        table(res[g], f"=== {title} ===", L)

    L.append("\n=== DRONE WINDOWS LOST TO THE STABILITY RULE ===")
    rejections = rejection_table(d, groups, thr, s_cut, L)
    L.append("\n=== FLICKER CHECK: active-box false alarms on a 50/60 Hz mains harmonic (+-2 Hz) ===")
    L.append("Reported only; the stability rule itself never looks at the frequency value.")
    flick = flicker_table(d, thr, combined, L)

    L.append("\n=== ACTIVE BACKGROUNDS: event counts vs the drone box (per sequence, per track) ===")
    L.append("cnt = median events per 0.5 s window. act/drone = active median / drone median.")
    matching = per_sequence_table(d, L)

    L.append("\n=== BOX WIDTH (all sequences incl. tune and development, full resolution, plain FFT) ===")
    L.append("Negatives in a width bin are the backgrounds of drones in that bin (same box size).")
    width_rec = width_analysis(d, thr, L)

    L.append("\n=== SIMULATED SMALLER SENSOR (heldout + challenging pooled, plain FFT, frozen threshold) ===")
    L.append("x, y and boxes integer-divided by 2 / 4; events on the same coarse pixel kept separate.")
    ds_delta = downsample_analysis(groups["heldout"] + groups["challenging"], thr, L)

    write_notes(res, matching, flick, rejections, width_rec, ds_delta, params, groups, d, thr, L)
    RESULTS_FILE.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
