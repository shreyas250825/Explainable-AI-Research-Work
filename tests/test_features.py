from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wireless_xai import FEATURE_NAMES, extract_features


def test_24_features_for_valid_iq_chunk():
    # Synthetic random IQ solely for testing extraction, not a wireless dataset.
    rng = np.random.default_rng(42)
    chunk = rng.integers(-100, 100, size=2048, dtype=np.int16)
    features = extract_features(chunk)
    assert tuple(features) == FEATURE_NAMES
    assert len(features) == 24
    assert np.all(np.isfinite(list(features.values())))
    assert features["energy"] > 0
    assert features["signal_power"] == pytest.approx(features["rms"]**2)
    assert features["dynamic_range"] == pytest.approx(features["peak_to_peak"])


def test_invalid_odd_length_rejected():
    with pytest.raises(ValueError):
        extract_features(np.arange(17, dtype=np.int16))


def test_int16_extrema_are_handled_without_integer_overflow():
    chunk = np.tile(np.array([-32768, 32767], dtype=np.int16), 8)
    features = extract_features(chunk)
    assert features["energy"] > 0
    assert np.isfinite(features["rms"])
