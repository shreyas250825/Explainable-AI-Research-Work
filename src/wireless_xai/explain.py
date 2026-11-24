"""Optional post-hoc SHAP example; not original paper's stored SHAP values."""

from __future__ import annotations
import numpy as np


def explain_with_deep_shap(model, training_features: np.ndarray,
                           evaluation_features: np.ndarray,
                           background_size: int = 100, evaluation_size: int = 50):
    """Return SHAP values using the approach in the source DNN notebook.

    The caller must provide *already standardized* training and evaluation
    arrays from a consistent fitted scaler. TensorFlow/SHAP compatibility and
    output shapes depend on installed versions; no attribution is fabricated.
    """
    import shap  # optional dependency
    training_features = np.asarray(training_features)
    evaluation_features = np.asarray(evaluation_features)
    if training_features.ndim != 2 or evaluation_features.ndim != 2:
        raise ValueError("Expected two-dimensional (samples, features) arrays")
    if training_features.shape[1] != evaluation_features.shape[1]:
        raise ValueError("Background and evaluation feature dimensions differ")
    background = training_features[:min(background_size, len(training_features))]
    examples = evaluation_features[:min(evaluation_size, len(evaluation_features))]
    if len(background) == 0 or len(examples) == 0:
        raise ValueError("Both arrays must contain at least one row")
    explainer = shap.DeepExplainer(model, background)
    return explainer.shap_values(examples)
