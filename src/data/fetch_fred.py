"""Download, decode and window every Phase 3 recording, identically on every machine.

    python -m src.data.fetch_fred            # everything in experiments/phase3_sequences.json
    python -m src.data.fetch_fred --only train/206 test/93

For each recording it downloads the zip from Hugging Face at the pinned revision,
keeps coordinates.txt, decodes Event/events.hdf5 into Event/events.npy, deletes
the zip and the .hdf5, and writes the Phase 1 window CSV if it is missing.
experiments/phase3_checksums.json records the SHA-256 of every events.npy and
CSV. The first machine writes it; every later run checks against it and stops on
a mismatch, so two people training on "the same data" really are.

Decoding needs Prophesee's ECF filter. On macOS and Linux build
third_party/hdf5_ecf (see its README) and run with
HDF5_PLUGIN_PATH=third_party/build. On Windows, src/data/ecf_convert.py uses
third_party/ecf/ecf.dll instead.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np

from src import phase1_fft as phase1
from src.data.ecf_convert import OUT_DTYPE

SPLIT = phase1.OUT_DIR / "phase3_sequences.json"
CHECKSUMS = phase1.OUT_DIR / "phase3_checksums.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(source, dest):
    """Copy CD/events into a structured .npy, checking that time never runs backwards."""
    try:
        file = h5py.File(source)
        events = file["CD/events"]
        events[:1]
    except OSError:
        from src.data.ecf_convert import convert          # Windows: ecf.dll
        convert(source, dest)
        return
    with file:
        out = np.lib.format.open_memmap(dest, mode="w+", dtype=OUT_DTYPE, shape=events.shape)
        last = -1
        for start in range(0, len(events), phase1.CHUNK):
            block = events[start:start + phase1.CHUNK]
            t = block["t"]
            if t[0] < last or np.any(np.diff(t) < 0):
                raise ValueError(f"{source}: event times decrease near row {start}")
            last = int(t[-1])
            for name in OUT_DTYPE.names:
                out[name][start:start + len(block)] = block[name]
        out.flush()
        del out


def fetch(seq, dataset, revision):
    from huggingface_hub import hf_hub_download
    started = time.monotonic()
    folder = phase1.seq_dir(seq)
    events = phase1.events_path(seq)
    if not events.exists():
        split, number = seq.split("/")
        with tempfile.TemporaryDirectory(dir=phase1.REPO / "data") as scratch:
            for attempt in range(3):
                try:
                    archive = hf_hub_download(dataset, f"{seq}.zip", repo_type="dataset",
                                              revision=revision, local_dir=scratch)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(10 * (attempt + 1))
            (folder / "Event").mkdir(parents=True, exist_ok=True)
            with ZipFile(archive) as zipped:
                for name in ("coordinates.txt", "Event/events.hdf5"):
                    with zipped.open(f"{number}/{name}") as source, (folder / name).open("wb") as dest:
                        shutil.copyfileobj(source, dest)
            partial = events.with_suffix(".partial.npy")
            decode(folder / "Event" / "events.hdf5", partial)
            partial.replace(events)
            (folder / "Event" / "events.hdf5").unlink()
    if not phase1.csv_path(seq).exists():
        phase1.run_sequence(seq)
    return seq, sha256(events), sha256(phase1.csv_path(seq)), time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    split = json.loads(SPLIT.read_text())
    wanted = args.only or [s for key in ("train", "validation", "test", "development") for s in split[key]]
    known = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch, s, split["dataset"], split["revision"]): s for s in wanted}
        for job in as_completed(jobs):
            try:
                seq, events_hash, csv_hash, seconds = job.result()
            except Exception as error:
                failures.append(jobs[job])
                print(f"FAILED {jobs[job]}: {error!r}", flush=True)
                continue
            entry = {"events.npy": events_hash, "csv": csv_hash}
            if seq in known and known[seq] != entry:
                failures.append(seq)
                print(f"MISMATCH {seq}: {entry} != {known[seq]}", flush=True)
                continue
            known[seq] = entry
            CHECKSUMS.write_text(json.dumps(dict(sorted(known.items())), indent=1) + "\n")
            print(f"ok {seq} ({seconds:.0f} s)", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} recordings failed: {failures}")


if __name__ == "__main__":
    main()
