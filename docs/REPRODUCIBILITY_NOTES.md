# Reproducibility notes and limitations

These notes are intended to prevent accidental overstatement of what the uploaded experimental notebooks establish.

1. **Historical numerical output.** The uploaded `Explanable_AI_DNN(1).ipynb` reports 292,955 processed rows; a stratified 20% test set of 58,591; 24 numeric feature columns; and a final test accuracy printed as `0.916796088218689` (approximately 91.68%). These are **historical notebook outputs**, not independently reproduced results.
2. **Global standardization.** In the source notebook `StandardScaler().fit_transform(X)` is called **before** `train_test_split`. Fitting preprocessing statistics on the whole table allows information from the evaluation partition into the preparation step.
3. **Test rows reused for validation.** The source model fits with `validation_data=(X_test, y_test)` while later evaluating on those same rows. Selection or monitoring during training may bias their interpretation as an untouched test set.
4. **Possible recording-level overlap.** Fixed-size chunks originate from long recordings, but the original processed CSV did not retain a recording identifier. Random row splits can mix correlated chunks from one recording between training and evaluation. The public extractor adds `source_file` to enable more conservative grouping.
5. **Chunk units.** `chunk_size=2048` in the original raw-data loader counted **int16 values**, corresponding to 1,024 complex I/Q pairs, not 2,048 pairs.
6. **Original loop boundary.** Some source extraction loops use `range(0, len(signal) - chunk_size, chunk_size)` and can skip the final complete chunk. This public implementation consumes each complete chunk. Resulting row counts need not match.
7. **Source feature semantics.** The original descriptor `zero_crossing_rate` uses the sign of signal **magnitude**, which is nonnegative. The feature should be interpreted with caution. Several source features are mathematically dependent (`dynamic_range` equals `peak_to_peak`; `signal_power` equals `rms²`; `variance` and `std` are related).
8. **SHAP compatibility.** The source notebook used SHAP `DeepExplainer` on TensorFlow/Keras. SHAP output format, model support and reproducibility depend on installed versions. Original SHAP attribution arrays are not supplied here.
9. **Data provenance.** Source capture hardware, precise receiver/sampling conditions, licenses and original `.data` files are not distributed in these uploads. Some aspects of the source workflow cannot therefore be reconstructed or independently audited.
10. **Revised example ≠ published claim.** `train_public_example.py` corrects several evaluation weaknesses, but any result from that script is a **new** run requiring separate reporting. Do not replace the manuscript's reported values with new figures without conducting and documenting the experiment.
11. **DQN status.** The exploratory DQN portion of `explanable_ai(1).ipynb` is summarized as pseudocode. No complete, verified DQN training run or learned weights are distributed here.

## Recommended release checklist

- Verify redistribution rights for any raw IQ files and derived data.
- Record raw recording IDs, capture parameters, timestamps, scenario information and class mappings.
- Decide whether the principal metric should be obtained from recording-disjoint (preferred) or another prespecified split.
- Fit scalers and select hyperparameters using training/validation only.
- Re-run SHAP with versioned model weights and an explicitly chosen background set.
- Link the exact dataset and partial code versions in the paper after confirming the links work publicly.
