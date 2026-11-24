# Dataset structure and public data schema

## What the uploads establish

The experimental notebooks read local files containing interleaved signed 16-bit IQ values. The original Google Drive working directory had `Wi-Fi/`, `LTE/` and `5G-NR/` top-level folders. Wi-Fi and LTE data appeared inside scenario folders; 5G-NR data were read directly beneath its folder. The exact acquisition instrumentation, usage rights, and raw dataset provenance are **not provided** by these notebooks.

## Supported local input layout

```text
Wireless/                    # Local folder you supply; not bundled here
├── Wi-Fi/
│   └── <scenario>/
│       └── <recording>.data
├── LTE/
│   └── <scenario>/
│       └── <recording>.data
└── 5G-NR/
    └── <recording>.data
```

Each `.data` is interpreted as sequential little-endian `int16` scalar values `I0,Q0,I1,Q1,...`. The public script assumes this ordering **because it is used by the source notebooks**; metadata must be checked before applying it to a different dataset. A chunk contains 2,048 scalar values (1,024 complex IQ pairs).

## Feature CSV columns

The optional script `scripts/prepare_features.py` writes:

- **24 numeric descriptors:** `mean`, `std`, `max`, `min`, `median`, `variance`, `rms`, `energy`, `skewness`, `kurtosis`, `peak_to_peak`, `crest_factor`, `phase_mean`, `phase_std`, `phase_variance`, `spectral_mean`, `spectral_std`, `spectral_max`, `spectral_energy`, `spectral_entropy`, `entropy`, `zero_crossing_rate`, `signal_power`, `dynamic_range`.
- **`network_type` (integer):** `0` Wi-Fi, `1` LTE, `2` 5G-NR.
- **`scenario` (text):** scenario folder (for 5G-NR, `5g-nr`).
- **`source_file` (text):** a relative recording identifier, added by the public script to permit grouped splitting. The original `new_processed_wireless_dataset.csv` loaded in the DNN notebook **did not contain** this additional column.

The notebook's cleaned data had 26 columns: 24 numeric features plus `network_type` and `scenario`, with 292,955 rows. This count documents the recorded run; it is **not** guaranteed by the public extractor.

## Uploading the dataset later

If the team has the right to redistribute records or processed features, provide a versioned file, size, SHA-256, source and collection metadata, license/permissions, and permanent link (GitHub Releases or a research-data repository). Update `data/README.md` and `docs/PAPER_AVAILABILITY_STATEMENT.md` at that time. Do not claim a dataset has been made public before it is actually uploaded and accessible.
