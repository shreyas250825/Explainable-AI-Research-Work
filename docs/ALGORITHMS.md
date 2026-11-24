# Algorithm descriptions

The following algorithms describe the disclosed portion of the work recorded in the two uploaded research notebooks. They are provided to make the public code understandable. Algorithm 3 explicitly describes a **proposed, safer evaluation protocol**, not the historical validation procedure reported in the original notebook.

## Algorithm 1 — IQ segmentation and 24-feature extraction

**Input:** path to an interleaved 16-bit IQ recording, class label `c ∈ {0, 1, 2}`, chunk length `L = 2048` *int16 scalar values*.  
**Output:** feature rows `z ∈ R^24` with class labels and optional source identifiers.

```text
for each labeled recording R:
    read R as an ordered stream of signed 16-bit values
    while L complete int16 values are available:
        chunk = next L values
        I = float64(chunk[0::2])
        Q = float64(chunk[1::2])
        x = I + j*Q
        magnitude = abs(x)
        phase = angle(x)
        spectrum = abs(FFT(magnitude))

        z = [time-domain statistics,
             higher-order moments,
             peak/amplitude descriptors,
             phase descriptors,
             spectral descriptors,
             magnitude and spectral entropy,
             source-notebook zero-crossing descriptor]

        if every entry in z is finite:
            write (z, class label, scenario, recording identifier)
    discard incomplete trailing chunk
```

**Source correspondence.** This uses the feature names in the `Experiment 4 - More Features Chunking` code block of `explanable_ai(1).ipynb`. The public streamer processes the final *complete* chunk, whereas some original notebook loops stopped one chunk early. It therefore does not promise identical record totals.

**Important signal convention:** `L=2048` means `1024` complex pairs. The feature labeled `zero_crossing_rate` in the notebook evaluates changes in the sign of **magnitude**, which is always nonnegative. It should not be interpreted as an I-channel zero-crossing rate.

## Algorithm 2 — Multi-class DNN model blueprint

**Input:** normalized feature vector `z ∈ R^24`.  
**Output:** a three-component class probability vector.

```text
h0 = z
h1 = Dropout( BatchNorm( ReLU( Dense(h0, 512) ) ), rate=0.30 )
h2 = Dropout( BatchNorm( ReLU( Dense(h1, 256) ) ), rate=0.30 )
h3 = Dropout( BatchNorm( ReLU( Dense(h2, 128) ) ), rate=0.20 )
h4 = ReLU( Dense(h3, 64) )
p  = Softmax( Dense(h4, 3) )
predicted_label = argmax(p)
```

Model compilation in the uploaded `Explanable_AI_DNN(1).ipynb`: Adam, sparse categorical cross-entropy, accuracy metric. The paper's three classes are **technology families**, not individual transmitter identities or a decoded protocol.

## Algorithm 3 — Public demonstration split and validation (corrected protocol)

The original notebook applied standardization before partitioning and reused its test rows as validation. The demonstrator intentionally changes that evaluation sequence:

```text
read features and corresponding class labels
remove missing / non-finite rows
if source_file identifies sufficient distinct recordings:
    partition by recording into disjoint train, validation, test sets
else:
    warn that row-based splitting is weaker
    perform stratified row-based splitting for demonstration

fit StandardScaler ONLY on training rows
transform training, validation, test using that same fitted scaler
build a new DNN using Algorithm 2
fit on training rows; monitor validation rows
perform a single final evaluation on held-out test rows
write newly calculated metrics (do not substitute into historical paper)
```

**Reporting rule:** the result of Algorithm 3 is a new experiment, not an independent replication of the source notebook's **91.68%** figure. Validating on unseen recording sessions or receivers is preferable to random chunk-level splitting.

## Algorithm 4 — Post-hoc SHAP analysis (illustrative)

```text
select a small background sample from the TRAINING partition
select a modest number of held-out evaluation rows
instantiate a SHAP explainer compatible with the trained model
calculate per-feature contributions to each class score
aggregate absolute contributions for a global ranking
interpret class-specific and per-record explanations cautiously
```

This is not a causal explanation. SHAP outputs depend on the reference distribution and can vary with correlated or redundant features. The original notebook used `shap.DeepExplainer`; the optional public wrapper exposes that same choice but must be tested against the installed SHAP/TensorFlow versions.

## Algorithm 5 — DQN comparison (summary only)

A separate experimental block in `explanable_ai(1).ipynb` implements a label-selection environment with actions `{0,1,2}` and rewards `+1` for a matching label and `−1` otherwise, using epsilon-greedy exploration and Q-value updates. The current release provides **documentation rather than a runnable DQN reconstruction**, because the exploratory notebook does not establish a complete reproducible reference training protocol (including a held-out recording-level evaluation). In a fixed-label problem, ordinary supervised classification is the primary baseline.
