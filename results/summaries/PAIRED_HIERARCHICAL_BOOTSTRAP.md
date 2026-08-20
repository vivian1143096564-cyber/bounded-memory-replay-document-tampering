# Paired hierarchical bootstrap

Replay 2k is compared with each baseline on identical frozen internal-test pages and three matched seeds. Each of 10,000 replicates resamples seeds and then pages within seeds. Values are Replay minus baseline in percentage points; negative values favor Replay for FAR and complete misses. Two-sided bootstrap p-values are descriptive because four methods and four metrics are examined.

| Baseline | Metric | Delta (pp) | 95% CI (pp) | p | Holm p |
|---|---|---:|---:|---:|---:|
| Direct FT | Macro F1 | +2.20 | [+1.03, +3.76] | 0.0000 | 0.0000 |
| Direct FT | Micro F1 | +1.83 | [+0.83, +3.20] | 0.0000 | 0.0000 |
| Direct FT | FAR >0.1% | -78.81 | [-83.17, -74.37] | 0.0000 | 0.0000 |
| Direct FT | Complete miss | +27.73 | [+21.36, +36.74] | 0.0000 | 0.0000 |
| LwF | Macro F1 | +1.77 | [+0.66, +3.01] | 0.0002 | 0.0014 |
| LwF | Micro F1 | +1.50 | [+0.46, +2.80] | 0.0006 | 0.0030 |
| LwF | FAR >0.1% | -75.94 | [-81.72, -69.62] | 0.0000 | 0.0000 |
| LwF | Complete miss | +21.71 | [+16.33, +26.16] | 0.0000 | 0.0000 |
| EWC | Macro F1 | +2.03 | [+0.88, +3.37] | 0.0000 | 0.0000 |
| EWC | Micro F1 | +1.72 | [+0.66, +3.08] | 0.0002 | 0.0014 |
| EWC | FAR >0.1% | -77.70 | [-82.11, -73.12] | 0.0000 | 0.0000 |
| EWC | Complete miss | +23.43 | [+17.09, +28.94] | 0.0000 | 0.0000 |
| Joint full | Macro F1 | +0.03 | [-0.13, +0.18] | 0.6700 | 1.0000 |
| Joint full | Micro F1 | -0.01 | [-0.17, +0.14] | 0.9894 | 1.0000 |
| Joint full | FAR >0.1% | -1.77 | [-5.15, +1.85] | 0.3474 | 1.0000 |
| Joint full | Complete miss | -0.09 | [-1.16, +1.07] | 0.8506 | 1.0000 |

Interpretation rule: a confidence interval excluding zero is evidence of a paired difference for that metric. This analysis does not authorize a global superiority claim, and multiplicity must be acknowledged.
