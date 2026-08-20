# JEI anonymous reproducibility package

This package supports the manuscript **Storage-Efficient Adaptation of Document Tampering Localization under Perceptual-Group-Disjoint Evaluation**.

## Scope and access policy

- The bundle contains code, fixed ID-level partitions, hard-negative IDs, run configurations, per-page derived metrics, and aggregate statistics.
- It does **not** contain RTM or DocTamper images, pretrained weights, adapted checkpoints, or third-party official code.
- The official RTM test set was not accessed and cannot be selected by the packaged target evaluator. `--split-file` is mandatory.
- Obtain RTM, DocTamper, the DTD implementation, and the public DTD checkpoint from their respective owners and follow their licenses.

## Directory map

- `code/`: training, strict target evaluation, DocTamper source evaluation, split construction, hard-negative mining, and statistics.
- `protocol/`: fixed perceptual groups, train/development/internal-test split, split summary, and hard-negative IDs.
- `configs/`: sanitized method configurations for all three seeds.
- `scripts/`: portable launcher for the five primary methods.
- `results/target_internal_test/`: per-page derived metrics and summaries for five methods × three seeds.
- `results/source_retention/`: FCD/SCD/TestingSet summaries.
- `results/summaries/`: manuscript tables and hierarchical-bootstrap outputs.

## Environment

The experiments used Python 3.10, PyTorch 2.7.1 with Intel XPU support, NumPy, Pillow, OpenCV, LMDB, and the dependencies required by the official DTD code. Hardware-specific installation should follow Intel Extension for PyTorch and the upstream DTD repository. No package in this archive redistributes third-party model or dataset assets.

## Reproduction order

1. Set `DATA_ROOT` to a directory containing the licensed datasets, official DTD code, and public weights.
2. Verify `protocol/perceptual_group_split.json` and `protocol/hard_negatives.json`.
3. Run a smoke test by adding a small `--max-train-steps` value to `scripts/run_train.sh` if needed.
4. Run each method sequentially for seeds `20260811`, `20260821`, and `20260831`:
   `bash scripts/run_train.sh replay 20260811`
5. Select checkpoints using development only.
6. Evaluate the frozen internal test once with `code/dtd_rtm_strict_eval.py --split-file protocol/perceptual_group_split.json --split-key internal_test`.
7. Evaluate source retention on FCD 2000, fixed SCD 1000, and fixed TestingSet 1000.
8. Run `python code/paired_hierarchical_bootstrap.py` after placing the per-page metric directories in the expected results layout.

## Evidence levels

- Primary confirmatory evidence: five methods, three seeds, frozen internal test.
- Secondary analyses: replay-memory size and random-split sensitivity.
- Development-only diagnostics are not part of the final method ranking.

## Citation and contact

For double-blind review this package intentionally omits author names and local machine paths. Contact and repository URL can be added after acceptance.
