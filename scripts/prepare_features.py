#!/usr/bin/env python3
"""Extract a 24-feature CSV from locally available interleaved-int16 recordings.

Expected input folder:
  raw/Wi-Fi/<scenario>/*.data
  raw/LTE/<scenario>/*.data
  raw/5G-NR/*.data

This script is a *cleaned partial reference implementation*, not a bit-for-bit
replay of the source notebooks (the notebook's original chunk loop omitted the
last full chunk in some recordings). Keep raw recordings out of public Git.
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wireless_xai import FEATURE_NAMES, extract_features

CLASS_LABELS = {"Wi-Fi": 0, "LTE": 1, "5G-NR": 2}


def iter_recordings(data_root: Path):
    """Yield (file, class, scenario) for the folder convention in the notebooks."""
    for technology, label in CLASS_LABELS.items():
        directory = data_root / technology
        if not directory.exists():
            continue
        if technology == "5G-NR":
            paths = sorted(directory.glob("*.data"))
        else:
            paths = sorted(directory.glob("*/*.data"))
        for path in paths:
            scenario = path.parent.name if technology != "5G-NR" else "5g-nr"
            yield path, label, scenario


def extract_csv(data_root: Path, output_csv: Path, chunk_size: int = 2048,
                max_chunks_per_file: int | None = None) -> tuple[int, int]:
    if chunk_size < 8 or chunk_size % 2:
        raise ValueError("chunk_size is a count of interleaved int16 values; use an even value >= 8")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    files_seen = written = 0
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FEATURE_NAMES, "network_type", "scenario", "source_file"])
        writer.writeheader()
        for path, label, scenario in iter_recordings(data_root):
            files_seen += 1
            with path.open("rb") as stream:
                chunk_count = 0
                while True:
                    if max_chunks_per_file is not None and chunk_count >= max_chunks_per_file:
                        break
                    chunk = np.fromfile(stream, dtype="<i2", count=chunk_size)
                    if len(chunk) < chunk_size:
                        break
                    chunk_count += 1
                    features = extract_features(chunk)
                    if not np.isfinite(list(features.values())).all():
                        continue
                    writer.writerow({**features, "network_type": label,
                                     "scenario": scenario,
                                     "source_file": str(path.relative_to(data_root))})
                    written += 1
    return files_seen, written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed_features.csv"))
    parser.add_argument("--chunk-size", type=int, default=2048, help="int16 values per chunk (2048 = 1024 IQ pairs)")
    parser.add_argument("--max-chunks-per-file", type=int, default=None,
                        help="Optional limit for quick demonstrations; omit for whole recording")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error(f"Not a data directory: {args.data_root}")
    files_seen, written = extract_csv(args.data_root, args.output, args.chunk_size, args.max_chunks_per_file)
    print(f"Processed {files_seen} recordings; wrote {written} valid feature rows to {args.output}")
    if not files_seen:
        print("Warning: no .data recordings found; see docs/DATA_DICTIONARY.md")


if __name__ == "__main__":
    main()
