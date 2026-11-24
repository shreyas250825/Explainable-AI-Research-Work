"""Partial public implementation of the 24 IQ-derived feature descriptors.

Adapted from the user's uploaded ``explanable_ai(1).ipynb`` notebook, whose
"Experiment 4 - More Features Chunking" block defines these descriptors.

Important: input is an interleaved *int16 scalar sequence* [I0,Q0,I1,Q1,...].
The source notebook's ``chunk_size=2048`` counts 2048 int16 values, i.e.
1024 complex IQ samples, not 2048 complex IQ samples.

This is research code, not a verified SDR decoder. In particular, the original
``zero_crossing_rate`` was calculated from the nonnegative magnitude and thus
is not a conventional signed zero-crossing rate. It is intentionally retained
here for schema compatibility with the notebook's feature table.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "mean", "std", "max", "min", "median", "variance", "rms", "energy",
    "skewness", "kurtosis", "peak_to_peak", "crest_factor", "phase_mean",
    "phase_std", "phase_variance", "spectral_mean", "spectral_std",
    "spectral_max", "spectral_energy", "spectral_entropy", "entropy",
    "zero_crossing_rate", "signal_power", "dynamic_range",
)


def extract_features(interleaved_iq: np.ndarray) -> dict[str, float]:
    """Return the 24 descriptors used in the later source notebook experiment.

    Accepts one 1-D array of alternating 16-bit I and Q values. Samples are
    converted to float64 *before* squaring to avoid int16 overflow. A chunk
    shorter than four complex samples is rejected. Zero-energy chunks can
    produce undefined higher-order moments; callers should drop nonfinite rows.
    """
    raw = np.asarray(interleaved_iq)
    if raw.ndim != 1 or len(raw) < 8 or len(raw) % 2:
        raise ValueError("Expected a 1-D interleaved IQ array with an even length >= 8")

    i = raw[0::2].astype(np.float64)
    q = raw[1::2].astype(np.float64)
    iq = i + 1j * q
    magnitude = np.abs(iq)
    phase = np.angle(iq)
    spectrum = np.abs(np.fft.fft(magnitude))
    eps = 1e-10
    rms = float(np.sqrt(np.mean(magnitude**2)))

    power_hist, _ = np.histogram(magnitude, bins=50)
    p_hist = (power_hist.astype(np.float64) + eps)
    p_hist /= p_hist.sum()
    p_spectrum = spectrum / (float(spectrum.sum()) + eps)

    features = {
        "mean": np.mean(magnitude),
        "std": np.std(magnitude),
        "max": np.max(magnitude),
        "min": np.min(magnitude),
        "median": np.median(magnitude),
        "variance": np.var(magnitude),
        "rms": rms,
        "energy": np.sum(magnitude**2),
        "skewness": pd.Series(magnitude).skew(),
        "kurtosis": pd.Series(magnitude).kurt(),
        "peak_to_peak": np.ptp(magnitude),
        "crest_factor": np.max(magnitude) / (rms + eps),
        "phase_mean": np.mean(phase),
        "phase_std": np.std(phase),
        "phase_variance": np.var(phase),
        "spectral_mean": np.mean(spectrum),
        "spectral_std": np.std(spectrum),
        "spectral_max": np.max(spectrum),
        "spectral_energy": np.sum(spectrum**2),
        "spectral_entropy": -np.sum(p_spectrum * np.log2(p_spectrum + eps)),
        "entropy": -np.sum(p_hist * np.log2(p_hist)),
        # The original notebook applies np.sign() to nonnegative |IQ|.
        "zero_crossing_rate": np.count_nonzero(np.diff(np.sign(magnitude))) / len(magnitude),
        "signal_power": np.mean(magnitude**2),
        "dynamic_range": np.ptp(magnitude),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}
