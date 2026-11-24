"""Public model architecture blueprint; no trained weights are distributed.

Architecture from ``Explanable_AI_DNN(1).ipynb`` (512/256/128/64 neurons).
The original notebook uses the test split as validation during training;
our *example training script* keeps validation and test partitions separate.
"""


def build_dnn(input_features: int = 24, num_classes: int = 3):
    """Create a new (untrained) Keras classifier consistent with the notebook."""
    if input_features < 1 or num_classes < 2:
        raise ValueError("input_features must be positive and num_classes >= 2")
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("Optional TensorFlow dependency required: pip install -r requirements-ml.txt") from exc

    layers = keras.layers
    model = keras.Sequential([
        layers.Input(shape=(input_features,)),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
