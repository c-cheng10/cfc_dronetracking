"""Plot the FFT of ONE 0.5 s window: a drone box next to background boxes.

For each drone we take the usable window whose prominence is closest to that
drone's median (a typical window, not the best one), and, at the same window,
the first shifted background box and the first active background box that are
usable there. Uses the locked settings of phase1_fft.py (0.5 ms bins,
50-1000 Hz).

Run from the repo root:
    python src/phase1_example_window.py                 # train/36
    python src/phase1_example_window.py --seq train/91
Output: experiments/phase1_e_single_window_example.png (train/36) or
        experiments/phase1_e_single_window_example_<split>_<seq>.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_fft as p1  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="train/36")
    seq = ap.parse_args().seq
    tracks, streams, results, _, _ = p1.run_sequence(seq, write_csv=False)

    fig, axes = plt.subplots(len(tracks), 1, figsize=(10, 4 * len(tracks)), squeeze=False)
    for ax, (i, tr) in zip(axes[:, 0], tracks.items()):
        mine = [(s, r) for s, r in zip(streams, results) if s["track_id"] == i]
        dr = next(r for s, r in mine if s["kind"] == "drone")
        if len(dr["prom"]) == 0:
            ax.set_title(f"ID {i} {tr['cls']}: no usable windows"); continue
        w = np.argmin(np.abs(dr["prom"] - np.median(dr["prom"])))
        start = dr["start_bin"][w]
        ax.plot(dr["freqs"], dr["spectra"][w], label=f"drone box (prominence {dr['prom'][w]:.1f})")
        for kind in ("shifted", "active"):
            # first background box of this kind that has the same window
            for s, r in mine:
                hit = np.flatnonzero(r["start_bin"] == start) if s["kind"] == kind else []
                if len(hit):
                    k = hit[0]
                    ax.plot(r["freqs"], r["spectra"][k], alpha=0.8,
                            label=f"{kind} background {s['name']} (prominence {r['prom'][k]:.1f})")
                    break
        ax.set_title(f"{seq} ID {i} {tr['cls']}: one window at t = {start / p1.FS:.2f} s")
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("FFT magnitude"); ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.tight_layout()
    split, num = seq.split("/")
    name = ("phase1_e_single_window_example.png" if seq == "train/36"
            else f"phase1_e_single_window_example_{split}_{num}.png")
    out = p1.OUT_DIR / name
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
