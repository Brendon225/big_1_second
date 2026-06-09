# Stage 1 ChemProt BioBART P2 Result Analysis

## Result Status

This run is valid as a first full ChemProt generative baseline.

Evidence:

- `model_dtype=float32`
- `parameter_dtype=torch.float32`
- `skipped=true` count is 0
- `valid_output_rate` is 1.0
- `relation_validity_rate` is 1.0
- test size is 4199, matching the full-run output rather than the 30-sample smoke setting

The previous all-zero result was caused by non-finite training loss and empty generation outputs. This run no longer has that failure mode.

## Main Metrics

Output directory:

`outputs/stage1/chemprot/biobart_text2text/P2_entity_type_description_seed42_full`

Core metrics:

- micro-F1: 0.7081
- macro-F1: 0.6743
- precision: 0.6993
- recall: 0.7172
- rare relation macro-F1: 0.6309
- dev loss: 0.1196
- test generation loss: 0.1356

Evaluation note:

The implemented ChemProt F1 excludes `NO_RELATION` from the positive-relation F1 calculation. `NO_RELATION` is still included in the confusion matrix and should be reported separately for diagnostic analysis.

## Training Dynamics

Training appears numerically stable.

- train loss count: 3200
- non-finite count: 0
- skipped count: 0
- dev loss by epoch:
  - epoch 1: 0.1991
  - epoch 2: 0.1646
  - epoch 3: 0.1211
  - epoch 4: 0.1226
  - epoch 5: 0.1196

The dev loss improves substantially from epoch 1 to epoch 3, then becomes nearly flat. Epoch 5 is slightly better than epoch 4, so this run does not show a severe overfitting signal from the available dev-loss trace.

## Per-Class Behavior

Positive-label per-class F1:

- CPR:3: 0.6252
- CPR:4: 0.7749
- CPR:5: 0.6178
- CPR:6: 0.6440
- CPR:9: 0.7095

Diagnostic `NO_RELATION` performance:

- precision: 0.5412
- recall: 0.4781
- F1: 0.5077

The strongest class is `CPR:4`, while `CPR:5` and `CPR:3` remain weaker. The negative class is not strong enough and causes many false positive relation predictions.

## Main Error Patterns

Top confusion patterns:

- `CPR:4 -> CPR:3`: 224
- `NO_RELATION -> CPR:4`: 149
- `CPR:9 -> NO_RELATION`: 115
- `NO_RELATION -> CPR:3`: 110
- `CPR:3 -> NO_RELATION`: 102
- `CPR:4 -> CPR:9`: 95
- `CPR:6 -> CPR:5`: 84

Interpretation:

1. Activation/inhibition confusion is still significant, especially `CPR:4 -> CPR:3`.
2. The model over-predicts positive relations for negative pairs.
3. `CPR:5` agonist and `CPR:6` antagonist are semantically close and show mutual confusion.
4. `CPR:9` sometimes collapses into `NO_RELATION`, suggesting that substrate/product cues are not always captured by the current P2 prompt.

## Research Interpretation

This result supports the Stage 1 pipeline in three ways:

1. The generative BioBART baseline can be trained and evaluated on full ChemProt.
2. The fixed output format `relation: <label>` is reliable after fp32 stabilization.
3. P2 entity-type-aware relation descriptions provide a usable baseline, but they do not solve fine-grained relation semantics.

This is exactly the gap that RSG-BioRE should target: relation labels are not merely output tokens; they need stable semantic prototypes and instance-prototype alignment to reduce confusion among semantically adjacent relation types.

## Next Experimental Actions

Recommended next runs:

1. Run P1, P2, and P3 BioBART full baselines under the same stable fp32 setting.
2. Add a `NO_RELATION` diagnostic metric to `metrics.json`.
3. Implement RSG-BioRE trainable prototype alignment and compare against this P2 baseline.
4. Add an error analysis table grouped by confusion pair, especially `CPR:4 -> CPR:3`, `CPR:6 -> CPR:5`, and `NO_RELATION -> positive`.

