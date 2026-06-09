# Stage 1 ChemProt BioBART Schema Comparison Analysis

## Run Validity

The three full BioBART ChemProt runs are valid for baseline comparison:

- P1: `relation_description`
- P2: `entity_type_aware_description`
- P3: `knowledge_enhanced_description`

All three runs use:

- model: `GanjinZero/biobart-base`
- backend: `hf`
- seed: 42
- epochs: 5
- train/dev/test: 5114/2991/4199
- `model_dtype=float32`
- `valid_output_rate=1.0`
- `relation_validity_rate=1.0`
- no skipped or non-finite training steps

This means the generated output format is no longer a bottleneck. The remaining errors are relation semantic classification errors.

## Main Results

| Method | Semantic Field | Micro-F1 | Macro-F1 | Precision | Recall | Rare-F1 | Overall Acc | NO_REL F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | relation_description | 0.7420 | 0.6997 | 0.7717 | 0.7146 | 0.6546 | 0.6935 | 0.5110 |
| P2 | entity_type_aware_description | 0.7081 | 0.6743 | 0.6993 | 0.7172 | 0.6309 | 0.6742 | 0.5077 |
| P3 | knowledge_enhanced_description | 0.7392 | 0.6943 | 0.7526 | 0.7262 | 0.6779 | 0.6930 | 0.5015 |

Ranking by micro-F1:

1. P1: 0.7420
2. P3: 0.7392
3. P2: 0.7081

P1 is slightly best overall. P3 is very close to P1 and has the best rare-relation F1. P2 is unexpectedly weaker than both P1 and P3.

## Per-Class F1

| Label | P1 | P2 | P3 |
|---|---:|---:|---:|
| CPR:3 | 0.6672 | 0.6252 | 0.6122 |
| CPR:4 | 0.8011 | 0.7749 | 0.8171 |
| CPR:5 | 0.6078 | 0.6178 | 0.6307 |
| CPR:6 | 0.7014 | 0.6440 | 0.7250 |
| CPR:9 | 0.7210 | 0.7095 | 0.6862 |

Interpretation:

- P1 is strongest on CPR:3 and CPR:9.
- P3 is strongest on CPR:4, CPR:5, and CPR:6.
- P2 does not win on any class, suggesting that the current entity-type-aware description adds little useful information for ChemProt because all relation candidates already share the same chemical-protein type pattern.

## Training Dynamics

P1 dev loss:

- epoch 1: 0.1983
- epoch 2: 0.1778
- epoch 3: 0.1223
- epoch 4: 0.1207
- epoch 5: 0.1312

P2 dev loss:

- epoch 1: 0.1991
- epoch 2: 0.1646
- epoch 3: 0.1211
- epoch 4: 0.1226
- epoch 5: 0.1196

P3 dev loss:

- epoch 1: 0.1905
- epoch 2: 0.1408
- epoch 3: 0.1350
- epoch 4: 0.1231
- epoch 5: 0.1246

P1 reaches its best dev loss at epoch 4, then slightly worsens at epoch 5. P2 has the best final dev loss but the weakest test F1, which indicates dev loss alone is not sufficient for method selection. P3 is stable and close to P1.

## Error Pattern Comparison

P1 major errors:

- `CPR:4 -> NO_RELATION`: 189
- `CPR:9 -> NO_RELATION`: 160
- `CPR:3 -> NO_RELATION`: 155
- `NO_RELATION -> CPR:4`: 153

P2 major errors:

- `CPR:4 -> CPR:3`: 224
- `NO_RELATION -> CPR:4`: 149
- `CPR:9 -> NO_RELATION`: 115
- `NO_RELATION -> CPR:3`: 110
- `CPR:6 -> CPR:5`: 84

P3 major errors:

- `CPR:9 -> NO_RELATION`: 168
- `NO_RELATION -> CPR:4`: 168
- `CPR:3 -> NO_RELATION`: 144
- `CPR:4 -> NO_RELATION`: 108
- `CPR:3 -> CPR:4`: 106

Key insight:

P2 reduces some false negatives into `NO_RELATION`, but introduces more positive-class semantic confusion, especially `CPR:4 -> CPR:3` and `CPR:6 -> CPR:5`. P1 and P3 are stronger overall but still struggle with deciding whether an entity pair expresses any target relation.

## Research Implications

These results are useful for the thesis because they show a clear limitation of pure prompt/schema-based generative extraction:

1. Richer descriptions do not monotonically improve performance.
2. Entity-type-aware wording is not automatically better when all candidate relations share the same entity-type signature.
3. Knowledge-enhanced descriptions help rare and fine-grained relation types but do not solve negative-vs-positive boundary errors.
4. The main remaining errors are semantic neighborhood errors, not output format errors.

This supports the motivation for RSG-BioRE. The next model should not only feed relation descriptions into the prompt, but should encode relation semantic texts into explicit prototypes and align instance representations with those prototypes. The current confusion patterns provide direct evidence for why prototype-level supervision is needed.

## Recommended Next Step

Move from prompt/schema comparison to the RSG-BioRE main method:

- Use P1/P2/P3 as baseline and ablation reference.
- Implement trainable relation semantic prototypes.
- Add instance-prototype alignment loss.
- Report prototype top-1/top-3 accuracy and generation-prototype agreement.
- Focus error analysis on:
  - `CPR:4` vs `CPR:3`
  - `CPR:5` vs `CPR:6`
  - positive relation vs `NO_RELATION`

