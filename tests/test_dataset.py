from pathlib import Path
import csv
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_features import extract_csv, iter_recordings


def test_labeling_and_streaming(tmp_path: Path):
    root = tmp_path / "raw"
    for tech, scenario in [("Wi-Fi", "wifi_scenario"), ("LTE", "lte_scenario"), ("5G-NR", "")]:
        path = root / tech / scenario
        path.mkdir(parents=True)
        # Two chunks + one leftover scalar. Two complete chunks should be included.
        np.arange(2048*2+1, dtype=np.int16).tofile(path / "sample.data")
    assert len(list(iter_recordings(root))) == 3
    destination = tmp_path / "features.csv"
    files_seen, rows = extract_csv(root, destination)
    assert files_seen == 3
    assert rows == 6
    with destination.open(newline="") as f:
        entries = list(csv.DictReader(f))
    assert {int(r["network_type"]) for r in entries} == {0, 1, 2}
    assert all(r["source_file"] for r in entries)
