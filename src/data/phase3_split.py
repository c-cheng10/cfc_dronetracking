"""Choose the Phase 3 recordings once and write experiments/phase3_sequences.json.

Run once: python -m src.data.phase3_split. It refuses to overwrite the file, so
everyone who pulls the repo trains and validates on the same recordings.

Rule, fixed before any Phase 3 result: from FRED recordings outside the
challenging split and not used before, draw per drone type 3 new training and
2 new validation recordings at random (seed 0). The DJI Mini 2 gets 6 new
training recordings, because it is the closest FRED type to the DJI Mini 4K we
will deploy against. A recording with two drone types counts towards whichever
of its types has the fewest eligible recordings and still needs one. The Phase 1
tune and heldout recordings stay in training and validation.
"""

import json

import numpy as np

from src import phase1_fft as phase1

TYPES = ("Betafpv air75", "DJI Mini 2", "DJI Mini 3", "DJI Tello EDU", "DarwinFPV cineape20")
NEW_TRAIN, NEW_VALIDATION = 3, 2
EXTRA_TRAIN = {"DJI Mini 2": 3}
DATASET = "GabrieleMagrini/FRED"
REVISION = "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd"


def main():
    out = phase1.OUT_DIR / "phase3_sequences.json"
    if out.exists():
        raise FileExistsError(f"{out} already exists; the split is fixed")
    data = phase1.REPO / "data" / "FRED"
    old = json.loads((phase1.OUT_DIR / "phase1_sequences.json").read_text())
    challenging = {line.strip().rstrip("/") for line in (data / "challenging_challenging_test_split.txt").read_text().splitlines() if line.strip()}
    used = {s for key in ("tune", "heldout", "challenging", "development") for s in old[key]}
    survey = json.loads((data / "survey.json").read_text())
    pool = [r for r in survey if r["path"][:-4] not in used and r["path"][:-4].split("/")[1] not in challenging]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pool))
    need = {t: NEW_TRAIN + EXTRA_TRAIN.get(t, 0) + NEW_VALIDATION for t in TYPES}
    available = {t: sum(t in r["classes"] for r in pool) for t in TYPES}
    picked = {t: [] for t in TYPES}
    for i in order:
        record = pool[i]
        wanted = [kind for kind in record["classes"] if kind in need and need[kind] > 0]
        if wanted:
            kind = min(wanted, key=available.__getitem__)
            picked[kind].append(record)
            need[kind] -= 1
    train, validation, sizes = list(old["tune"]), list(old["heldout"]), {}
    for kind, records in picked.items():
        n_train = min(NEW_TRAIN + EXTRA_TRAIN.get(kind, 0), len(records) - min(NEW_VALIDATION, len(records) // 2))
        for j, record in enumerate(records):
            seq = record["path"][:-4]
            sizes[seq] = record["size_mb"]
            (train if j < n_train else validation).append(seq)
    split = {
        "_note": "Phase 3 split, chosen by src/data/phase3_split.py before any Phase 3 result. "
                 "test is FRED's challenging split: score it once, after the models stop changing. "
                 "development (train/36) is never counted as a result.",
        "dataset": DATASET, "revision": REVISION,
        "train": train, "validation": validation,
        "test": sorted(old["challenging"]), "development": old["development"],
        "download_mb": sizes,
        "shortfall": {k: v for k, v in need.items() if v},
    }
    out.write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps({k: split[k] for k in ("train", "validation", "shortfall")}, indent=1))
    print(f"new download: {sum(sizes.values()) / 1000:.1f} GB in {len(sizes)} recordings")


if __name__ == "__main__":
    main()
