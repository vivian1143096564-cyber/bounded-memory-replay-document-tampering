# Supplementary Experiment Audit

## Three-seed stability

|Metric|Mean|Sample SD|Range|
|---|---:|---:|---:|
|macro_f1|0.071900|0.001638|0.070926-0.073791|
|micro_f1|0.065949|0.001709|0.064718-0.067901|
|macro_iou|0.053104|0.001158|0.052290-0.054430|
|far_gt_01pct|0.203091|0.021288|0.178808-0.218543|
|complete_miss|0.316204|0.027440|0.291667-0.345833|
|mean_retention|0.992046|0.008056|0.985540-1.001057|

## Replay memory ablation

|Memory|Macro F1|Micro F1|FAR >0.1%|Complete miss|Mean source retention|
|---:|---:|---:|---:|---:|---:|
|500|0.0724|0.0676|0.2517|0.2639|0.9946|
|1000|0.0728|0.0671|0.2053|0.3403|0.9946|
|2000|0.0738|0.0679|0.2185|0.2917|1.0011|

## Split comparison

Strict perceptual-group split macro F1: 0.0738; random image split macro F1: 0.0804.
Strict FAR: 0.2185; random FAR: 0.2649.
Random split audit: {'train': 4063, 'development': 871, 'internal_test': 869}; cross-partition ID overlap = 0; retained train-only hard negatives = 567; official RTM test read = false.