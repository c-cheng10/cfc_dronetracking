"""Plot the FFT of ONE 0.5 s window: a drone box next to a background box.

For each drone we take the usable window whose prominence is closest to that
drone's median (a typical window, not the best one), and the background box
window at the same time. Uses the default settings of phase1_fft.py
(0.5 ms bins, 50-1000 Hz).

Run from the repo root:
    python src/phase1_example_window.py
Output: experiments/phase1_e_single_window_example.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_fft as p1  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    tracks = p1.load_tracks(p1.SEQ_DIR / "coordinates.txt")
    ev = np.load(p1.EVENTS, mmap_mode="r")
    n_bins = int(ev["t"][-1]) // p1.BIN_US + 1

    fig, axes = plt.subplots(len(tracks), 1, figsize=(10, 4 * len(tracks)))
    for ax, (i, tr) in zip(np.atleast_1d(axes), tracks.items()):
        offsets, _ = p1.choose_background_offsets(tracks, i)
        boxes = [(tr, 0, 0)] + [(tr, dx, dy) for dx, dy in offsets]
        counts = p1.count_events(ev, boxes, n_bins)
        res = []
        for (b, dx, dy), c, bg in zip(boxes, counts, [False] + [True] * len(offsets)):
            vb = p1.valid_bin_mask(b, dx, dy, tracks, i, n_bins, bg)
            res.append(p1.window_spectra(c, vb))
        t0, _, prom, usable, spectra, freqs = res[0]
        cand = np.flatnonzero(usable)
        w = cand[np.argmin(np.abs(prom[cand] - np.median(prom[cand])))]
        # first background box that is usable in the same window
        bg = next((r for r in res[1:] if r[3][w]), None)
        ax.plot(freqs, spectra[w], label=f"drone box (prominence {prom[w]:.1f})")
        if bg is not None:
            ax.plot(freqs, bg[4][w], label=f"background box (prominence {bg[2][w]:.1f})", alpha=0.8)
        ax.set_title(f"{p1.DRONE_NAMES[i]}: one window at t = {t0[w]:.2f} s")
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("FFT magnitude"); ax.legend()
    fig.tight_layout()
    out = p1.OUT_DIR / "phase1_e_single_window_example.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
