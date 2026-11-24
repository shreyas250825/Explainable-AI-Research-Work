# Technical methodology

## Scope

The task is three-way physical-layer technology-family classification: Wi-Fi (`0`), LTE (`1`), and 5G-NR (`2`). It is not decoding of PHY/MAC frames, identification of a radio manufacturer, or exact modulation-order prediction.

## IQ representation

For a chunk of `L = 2048` interleaved signed 16-bit values, even-indexed values form `I[n]` and odd-indexed values form `Q[n]`, giving `N = L/2 = 1024` complex samples:

\[
x[n] = I[n] + j Q[n], \quad a[n] = |x[n]|, \quad \phi[n] = \arg x[n].
\]

All feature operations are performed on floating-point conversions of `I` and `Q`. This avoids integer overflow when forming squared-energy descriptors.

## Feature groups

| Family | Feature names in the notebook |
| --- | --- |
| Magnitude statistics | `mean`, `std`, `max`, `min`, `median`, `variance` |
| Energy / power | `rms`, `energy`, `signal_power` |
| Higher moments | `skewness`, `kurtosis` |
| Peak descriptors | `peak_to_peak`, `crest_factor`, `dynamic_range` |
| Phase | `phase_mean`, `phase_std`, `phase_variance` |
| Fourier magnitude | `spectral_mean`, `spectral_std`, `spectral_max`, `spectral_energy` |
| Entropy | `spectral_entropy`, `entropy` |
| Legacy notebook descriptor | `zero_crossing_rate` |

Total: **24 features**. Some descriptors are algebraically redundant: `variance` is related to `std`, `signal_power` to `rms`, and `dynamic_range` equals `peak_to_peak` in the source notebook. Such feature dependence matters when interpreting SHAP rankings.

The Fourier features come from the FFT **of the magnitude series**, not directly from the original complex IQ samples. The feature named `zero_crossing_rate` was applied to magnitude; because `a[n] ≥ 0`, it is not a conventional signed zero-crossing metric. See [reproducibility notes](REPRODUCIBILITY_NOTES.md).

## Classification network

The released module `src/wireless_xai/dnn.py` instantiates the DNN architecture recorded in the second notebook: a 24-dimensional input and fully connected layers of 512, 256, 128 and 64 ReLU units, with batch normalization and dropout before a three-unit softmax output.

For class `c`, a model predicts `p(c | z)` and selects the label with maximum probability. Training uses sparse categorical cross-entropy and Adam. No trained weights are distributed in this release.

## Interpretability

The source notebook calls `shap.DeepExplainer` with a small background sample and displays summary and feature-importance bar plots for evaluation examples. Those figures demonstrate the *intended* explanation procedure; the public code contains no original SHAP arrays. A ranking of correlated descriptors should not be interpreted as physical causation.

## Evaluation boundary

A cleaned code demonstration first partitions recordings, fits feature scaling only on training data, and then trains with a separate validation set. This differs from the original notebook protocol. For publication-quality evaluation, recording identifiers, split grouping, acquisition metadata, and results from the revised evaluation must be retained.
