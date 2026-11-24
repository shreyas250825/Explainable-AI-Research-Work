#!/usr/bin/env python3
"""Optional public training example for a *new* model; not the published result.

Requires locally available processed 24-feature CSV and TensorFlow. The
training/validation/test setup intentionally avoids the two major evaluation
issues in the supplied notebook: scaling before splitting and using the test
set to monitor training. When recording IDs are available, groups are split so
related chunks never span splits. No performance figure is asserted here.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wireless_xai import FEATURE_NAMES
from wireless_xai.dnn import build_dnn


def split_indices(df: pd.DataFrame, *, seed: int = 42):
    """Return disjoint train/validation/test row indices; use groups if present."""
    indices = np.arange(len(df))
    if "source_file" in df and df["source_file"].notna().all() and df["source_file"].nunique() >= 10:
        groups = df["source_file"].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_val_rel, test_rel = next(splitter.split(indices, groups=groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed+1)
        train_inner, val_inner = next(inner.split(train_val_rel, groups=groups[train_val_rel]))
        return train_val_rel[train_inner], train_val_rel[val_inner], test_rel, True

    print("WARNING: no suitable recording-level source_file groups; row-based split is only a demonstration.")
    y = df["network_type"].to_numpy()
    train_val, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.2, stratify=y[train_val], random_state=seed+1)
    return train, val, test, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Processed features CSV (not bundled)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("results/public_example"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = [*FEATURE_NAMES, "network_type"]
    missing = [name for name in required if name not in df]
    if missing:
        parser.error(f"CSV missing required columns: {missing}")
    df = df.dropna(subset=required).drop_duplicates(subset=required)
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    if set(df["network_type"].astype(int).unique()) != {0, 1, 2}:
        parser.error("Expected class labels 0, 1, 2 for Wi-Fi, LTE, 5G-NR")

    tr, va, te, grouped = split_indices(df)
    y = df["network_type"].to_numpy().astype(int)
    for name, index in [("training", tr), ("validation", va), ("test", te)]:
        if set(np.unique(y[index])) != {0, 1, 2}:
            parser.error(f"The {name} split does not include all classes; add more recordings or adjust split")

    X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X[tr])  # fit on training data ONLY
    X_train, X_val, X_test = [scaler.transform(X[ix]) for ix in (tr, va, te)]

    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    model = build_dnn(input_features=len(FEATURE_NAMES))
    model.fit(X_train, y[tr], validation_data=(X_val, y[va]),
              epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    predicted = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y[te], predicted, labels=[0, 1, 2],
                                   target_names=["Wi-Fi", "LTE", "5G-NR"], output_dict=True, zero_division=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "New public example run, not the original manuscript experiment",
        "group_split": grouped,
        "rows": {"train": len(tr), "validation": len(va), "test": len(te)},
        "test_accuracy": accuracy_score(y[te], predicted),
        "confusion_matrix": confusion_matrix(y[te], predicted, labels=[0, 1, 2]).tolist(),
        "classification_report": report,
    }
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path} (new experiment, not notebook result)")


if __name__ == "__main__":
    main()
