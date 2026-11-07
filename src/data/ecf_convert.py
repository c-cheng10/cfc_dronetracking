"""Convert a FRED events.hdf5 (Prophesee ECF compression) into a plain events.npy.

h5py can't decompress ECF, so we read each compressed chunk as raw bytes and
decode it with third_party/ecf/ecf.dll.

Usage, from the repo root:
    python -m src.data.ecf_convert data/FRED/train/36/36/Event/events.hdf5

Output: events.npy next to the input, with fields x, y, p, t (t in microseconds).
Open it without loading it all into RAM:
    ev = np.load(path, mmap_mode="r")
"""
import argparse
import ctypes
from pathlib import Path

import h5py
import numpy as np

ECF_FILTER_ID = 36559
OUT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("t", "<u4")])
DLL_PATH = Path(__file__).resolve().parents[2] / "third_party" / "ecf" / "ecf.dll"


def load_decoder():
    if not DLL_PATH.exists():
        raise FileNotFoundError(f"{DLL_PATH} not found. See third_party/ecf/README.md to build it.")
    dll = ctypes.CDLL(str(DLL_PATH))
    dll.ecf_num_events.argtypes = [ctypes.c_char_p]
    dll.ecf_num_events.restype = ctypes.c_longlong
    dll.ecf_decode.argtypes = [ctypes.c_char_p, ctypes.c_longlong, ctypes.c_void_p]
    dll.ecf_decode.restype = ctypes.c_longlong
    return dll


def convert(src, dst=None):
    src = Path(src)
    dst = Path(dst) if dst else src.with_name("events.npy")
    dll = load_decoder()

    with h5py.File(src, "r") as f:
        ds = f["CD/events"]
        plist = ds.id.get_create_plist()
        filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
        if filters != [ECF_FILTER_ID]:
            raise ValueError(f"expected only the ECF filter, found {filters}")

        total = ds.shape[0]
        chunk = ds.chunks[0]
        width, height = (int(v) for v in f.attrs["geometry"].split("x"))
        indexes = f["CD/indexes"][:]

        out = np.lib.format.open_memmap(dst, mode="w+", dtype=OUT_DTYPE, shape=(total,))
        written = 0
        last_t = -1
        for start in range(0, total, chunk):
            mask, raw = ds.id.read_direct_chunk((start,))
            if mask != 0:
                raise ValueError(f"chunk at {start} skipped its filter (mask {mask})")
            n = dll.ecf_num_events(raw)
            buf = np.empty(n, dtype=ds.dtype)
            if dll.ecf_decode(raw, len(raw), buf.ctypes.data) != n:
                raise ValueError(f"decoder failed on chunk at {start}")

            # Sanity checks: a bad decode shows up as impossible values.
            if n != min(chunk, total - start):
                raise ValueError(f"chunk at {start} holds {n} events")
            if buf["x"].max() >= width or buf["y"].max() >= height:
                raise ValueError(f"pixel outside the sensor in chunk at {start}")
            if not np.isin(buf["p"], (0, 1)).all():
                raise ValueError(f"bad polarity in chunk at {start}")
            if buf["t"][0] < last_t or (np.diff(buf["t"]) < 0).any():
                raise ValueError(f"time goes backwards in chunk at {start}")
            if buf["t"][-1] >= 2**32:
                raise ValueError("timestamp too large for uint32")
            last_t = int(buf["t"][-1])

            for name in OUT_DTYPE.names:
                out[name][start:start + n] = buf[name]
            written += n

        if written != total:
            raise ValueError(f"wrote {written} of {total} events")

        # Independent check: the file's own index says event `id` is at or after time `ts`.
        rows = indexes[(indexes["ts"] >= 0) & (indexes["id"] < total)]
        bad = int((out["t"][rows["id"].astype(np.int64)] < rows["ts"]).sum())
        if bad:
            raise ValueError(f"{bad} of {len(rows)} index rows disagree with the decoded times")

        out.flush()
        print(f"{dst}: {total:,} events, {last_t / 1e6:.2f} s, "
              f"sensor {width}x{height}, {len(rows):,} index rows match")
    return dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src")
    parser.add_argument("--dst")
    args = parser.parse_args()
    convert(args.src, args.dst)
