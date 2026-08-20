# Submission Evidence Statistics

## Perceptual-group protocol

- **hash:** 64-bit difference hash (dHash)
- **edge rule:** Hamming distance <= 4
- **group rule:** connected components; every component assigned wholly to one partition
- **allocation:** 70/15/15 train/development/internal test
- **seed:** 20260811
- **pages:** 5803
- **groups:** 3966
- **multi image groups:** 1273
- **largest group:** 38
- **cross partition group violations:** 0
- **official test read:** False
- **internal test tampered pages:** 720
- **internal test pristine pages:** 151
- **internal gt area ratio median:** 0.005656886190239293
- **internal gt area ratio q1:** 0.002329413284571721
- **internal gt area ratio q3:** 0.01254046498428579
- **internal gt area ratio min:** 8.184058557398756e-05
- **internal gt area ratio max:** 0.3018563357546408
- **naming warning:** This split is perceptual-group-disjoint, not verified document-source-disjoint.

## Dataset partitions

|Partition|Pages|Pristine|Cover|Copy-move|Edit|Inpaint|Insert|Splice|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|train|4061|704|577|1667|133|439|197|344|
|development|871|151|124|357|29|94|42|74|
|internal_test|871|151|124|357|29|94|42|74|

## Page-bootstrap 95% confidence intervals

10,000 page-level percentile-bootstrap resamples; seed 20260804.

|Method|Metric|Estimate|95% CI|N|
|---|---|---:|---:|---:|
|Direct FT|macro_precision|2.87%|[2.43%, 3.33%]|720|
|Direct FT|macro_recall|13.73%|[12.11%, 15.44%]|720|
|Direct FT|macro_f1|3.55%|[3.06%, 4.05%]|720|
|Direct FT|macro_iou|1.94%|[1.67%, 2.24%]|720|
|Direct FT|pristine_far_gt_0.1pct|100.00%|[100.00%, 100.00%]|151|
|Direct FT|complete_miss|0.28%|[0.00%, 0.69%]|720|
|LwF|macro_precision|4.13%|[3.47%, 4.84%]|720|
|LwF|macro_recall|12.02%|[10.39%, 13.73%]|720|
|LwF|macro_f1|4.39%|[3.73%, 5.09%]|720|
|LwF|macro_iou|2.52%|[2.12%, 2.97%]|720|
|LwF|pristine_far_gt_0.1pct|99.34%|[98.01%, 100.00%]|151|
|LwF|complete_miss|0.28%|[0.00%, 0.69%]|720|
|EWC|macro_precision|3.46%|[2.93%, 4.04%]|720|
|EWC|macro_recall|12.25%|[10.62%, 13.96%]|720|
|EWC|macro_f1|3.98%|[3.39%, 4.61%]|720|
|EWC|macro_iou|2.25%|[1.90%, 2.63%]|720|
|EWC|pristine_far_gt_0.1pct|100.00%|[100.00%, 100.00%]|151|
|EWC|complete_miss|0.14%|[0.00%, 0.42%]|720|
|Joint full|macro_precision|10.72%|[8.74%, 12.72%]|720|
|Joint full|macro_recall|10.48%|[8.71%, 12.33%]|720|
|Joint full|macro_f1|7.21%|[5.84%, 8.66%]|720|
|Joint full|macro_iou|5.29%|[4.15%, 6.47%]|720|
|Joint full|pristine_far_gt_0.1pct|24.50%|[17.88%, 31.79%]|151|
|Joint full|complete_miss|29.86%|[26.53%, 33.06%]|720|
|Replay 500|macro_precision|10.94%|[8.95%, 12.90%]|720|
|Replay 500|macro_recall|10.51%|[8.75%, 12.34%]|720|
|Replay 500|macro_f1|7.24%|[5.86%, 8.69%]|720|
|Replay 500|macro_iou|5.30%|[4.17%, 6.48%]|720|
|Replay 500|pristine_far_gt_0.1pct|25.17%|[18.54%, 32.45%]|151|
|Replay 500|complete_miss|26.39%|[23.19%, 29.58%]|720|
|Replay 1000|macro_precision|10.77%|[8.75%, 12.77%]|720|
|Replay 1000|macro_recall|9.98%|[8.24%, 11.80%]|720|
|Replay 1000|macro_f1|7.28%|[5.88%, 8.75%]|720|
|Replay 1000|macro_iou|5.39%|[4.23%, 6.60%]|720|
|Replay 1000|pristine_far_gt_0.1pct|20.53%|[14.57%, 27.15%]|151|
|Replay 1000|complete_miss|34.03%|[30.56%, 37.50%]|720|
|Replay 2000|macro_precision|11.48%|[9.45%, 13.54%]|720|
|Replay 2000|macro_recall|10.47%|[8.71%, 12.32%]|720|
|Replay 2000|macro_f1|7.38%|[5.98%, 8.86%]|720|
|Replay 2000|macro_iou|5.44%|[4.29%, 6.66%]|720|
|Replay 2000|pristine_far_gt_0.1pct|21.85%|[15.23%, 29.14%]|151|
|Replay 2000|complete_miss|29.17%|[25.97%, 32.50%]|720|
