# Stage 1 ChemProt RSG-BioRE R2 Result Analysis

## Run Status

The RSG-BioRE R2 full run is valid but underperforms the BioBART schema baselines.

Evidence that the run is technically valid:

- `backend`: `hf_rsg`
- train/dev/test: 5114/2991/4199
- `valid_output_rate`: 1.0
- `relation_validity_rate`: 1.0
- `model_dtype=float32`
- `parameter_dtype=torch.float32`
- skipped/non-finite training steps: 0
- prototype outputs were produced:
  - `prototype_scores.jsonl`
  - `prototype_analysis.csv`

Therefore, this is not a formatting or numerical-collapse failure. It is a modeling failure of the current RSG design.

## Main Comparison

| Method | Micro-F1 | Macro-F1 | Rare-F1 | Overall Acc | NO_REL F1 |
|---|---:|---:|---:|---:|---:|
| BioBART P1 | 0.7420 | 0.6997 | 0.6546 | 0.6935 | 0.5110 |
| BioBART P2 | 0.7081 | 0.6743 | 0.6309 | 0.6742 | 0.5077 |
| BioBART P3 | 0.7392 | 0.6943 | 0.6779 | 0.6930 | 0.5015 |
| RSG-BioRE R2 | 0.6997 | 0.6432 | 0.5885 | 0.6609 | 0.4729 |

RSG-BioRE R2 is worse than:

- P1 by 4.23 micro-F1 points
- P2 by 0.84 micro-F1 points
- P3 by 3.95 micro-F1 points

This means the first real RSG implementation is not yet a successful main method.

## Prototype Diagnostics

Prototype metrics:

- prototype top-1 accuracy: 0.6516
- prototype top-3 accuracy: 0.9514
- generation-prototype agreement: 0.8926

Interpretation:

The prototype branch is not random. It ranks the gold label within top 3 for most samples, but it often fails to put the gold label at rank 1. The high generation-prototype agreement means generation and prototype branches make many of the same decisions. The prototype branch is currently reinforcing the generator's bias rather than correcting it.

Gold-label rank distribution in prototype scores:

- rank 1: 2736
- rank 2: 870
- rank 3: 389
- rank 4: 136
- rank 5: 45
- rank 6: 23

This suggests that the prototype space has useful coarse semantic structure, but the decision margin is not strong enough for fine-grained top-1 classification.

## Per-Class Behavior

| Label | Generation Recall | Prototype Top-1 Recall | Prototype Top-3 Recall |
|---|---:|---:|---:|
| CPR:3 | 0.5542 | 0.5768 | 0.9864 |
| CPR:4 | 0.8130 | 0.8130 | 0.9704 |
| CPR:5 | 0.7514 | 0.6973 | 0.9676 |
| CPR:6 | 0.5973 | 0.5597 | 0.9249 |
| CPR:9 | 0.6289 | 0.5450 | 0.8416 |
| NO_RELATION | 0.4503 | 0.4781 | 0.9788 |

The prototype branch helps candidate retrieval but does not help final discrimination. CPR:9 and NO_RELATION remain especially weak.

## Major Error Patterns

Top generation confusions:

- `NO_RELATION -> CPR:4`: 200
- `CPR:9 -> NO_RELATION`: 149
- `CPR:3 -> CPR:4`: 137
- `CPR:3 -> NO_RELATION`: 111
- `CPR:4 -> CPR:3`: 103
- `NO_RELATION -> CPR:3`: 100
- `CPR:6 -> CPR:5`: 91

Top prototype top-1 confusions are similar:

- `NO_RELATION -> CPR:4`: 196
- `CPR:9 -> NO_RELATION`: 171
- `CPR:3 -> CPR:4`: 135
- `CPR:4 -> CPR:3`: 108
- `CPR:6 -> CPR:5`: 102

The similarity between generation errors and prototype errors explains why R2 does not improve the final relation extraction metric.

## Likely Root Causes

1. Prototype alignment is only an auxiliary loss.
   The decoder is not explicitly guided by prototype scores at inference time.

2. The prototype encoder and instance encoder share the same BioBART encoder.
   This creates a coupled representation space, but not necessarily a discriminative prototype space.

3. Top-3 prototype accuracy is high, but top-1 is modest.
   The method retrieves plausible relation candidates, but does not sharpen decision boundaries.

4. `lambda=0.1` may be too weak or poorly scheduled.
   Alignment loss remains large relative to generation loss and may not converge enough.

5. The current instance representation may not emphasize the marked entity pair strongly enough.
   Many ChemProt documents contain multiple chemical-protein pairs in the same text, so pooling over the whole prompt can blur pair-specific semantics.

## Research Interpretation

This result should not be framed as the final RSG-BioRE method. It should be treated as R2-alpha:

- It verifies that the RSG training and output pipeline works.
- It proves prototype diagnostics can be produced.
- It shows prototype top-3 retrieval has signal.
- It also shows that simple auxiliary alignment is insufficient.

The next method version should make the prototype signal operationally useful for generation or classification, rather than only adding it as a weak regularizer.

## Recommended Next Version

R2.1 should focus on making prototypes discriminative before adding a guided decoder:

1. Add explicit entity markers to the input text.
2. Use the encoder representation of entity marker positions instead of mean pooling over the entire prompt.
3. Add a warm-up schedule for alignment loss:
   - early epochs: small lambda
   - later epochs: larger lambda
4. Try `alignment_lambda` values: 0.02, 0.05, 0.1.
5. Add a prototype-only auxiliary classifier metric during dev evaluation.
6. Consider inference-time interpolation:
   - generation label probability
   - prototype similarity score

R3 can then add guided decoding or score fusion after R2.1 produces stronger prototype top-1 accuracy.

