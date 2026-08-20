from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEEDS = ("", "_seed21", "_seed31")
METHODS = {
    "Direct FT": "compute_matched_direct",
    "LwF": "lwf",
    "EWC": "ewc",
    "Joint full": "joint_full_source",
}
REPLAY = "mixed_replay"
N_BOOT = 10_000
RNG_SEED = 20260816


def load(method: str, suffix: str) -> dict[str, dict[str, float | str]]:
    path = ROOT / f"{method}{suffix}_internal_test" / "sample_metrics.csv"
    rows: dict[str, dict[str, float | str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok":
                continue
            rows[row["image_id"]] = {
                "is_tampered": float(row["is_tampered"]),
                "f1": float(row["f1"]),
                "tp": float(row["tp"]),
                "fp": float(row["fp"]),
                "fn": float(row["fn"]),
                "pred_area_ratio": float(row["pred_area_ratio"]),
            }
    return rows


def metrics(rows: list[dict[str, float | str]]) -> np.ndarray:
    tampered = [r for r in rows if r["is_tampered"] == 1.0]
    pristine = [r for r in rows if r["is_tampered"] == 0.0]
    macro = np.mean([float(r["f1"]) for r in tampered])
    tp = sum(float(r["tp"]) for r in tampered)
    fp = sum(float(r["fp"]) for r in tampered)
    fn = sum(float(r["fn"]) for r in tampered)
    micro = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    far = np.mean([float(r["pred_area_ratio"]) > 0.001 for r in pristine])
    miss = np.mean([(float(r["tp"]) == 0.0) for r in tampered])
    return np.array([macro, micro, far, miss], dtype=np.float64)


def paired_comparison(name: str, method: str, rng: np.random.Generator):
    paired_by_seed = []
    point_deltas = []
    for suffix in SEEDS:
        replay = load(REPLAY, suffix)
        baseline = load(method, suffix)
        ids = sorted(set(replay) & set(baseline))
        if len(ids) != len(replay) or len(ids) != len(baseline):
            raise RuntimeError(f"Unmatched pages for {name}{suffix}: {len(ids)}")
        replay_rows = [replay[i] for i in ids]
        baseline_rows = [baseline[i] for i in ids]
        paired_by_seed.append((replay_rows, baseline_rows))
        point_deltas.append(metrics(replay_rows) - metrics(baseline_rows))

    point = np.mean(point_deltas, axis=0)
    boot = np.empty((N_BOOT, 4), dtype=np.float64)
    for b in range(N_BOOT):
        seed_indices = rng.integers(0, len(SEEDS), len(SEEDS))
        deltas = []
        for seed_idx in seed_indices:
            replay_rows, baseline_rows = paired_by_seed[int(seed_idx)]
            page_indices = rng.integers(0, len(replay_rows), len(replay_rows))
            r_sample = [replay_rows[int(i)] for i in page_indices]
            b_sample = [baseline_rows[int(i)] for i in page_indices]
            deltas.append(metrics(r_sample) - metrics(b_sample))
        boot[b] = np.mean(deltas, axis=0)

    low = np.percentile(boot, 2.5, axis=0)
    high = np.percentile(boot, 97.5, axis=0)
    p = 2.0 * np.minimum(np.mean(boot <= 0.0, axis=0), np.mean(boot >= 0.0, axis=0))
    p = np.minimum(p, 1.0)
    return name, point, low, high, p


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    results = [paired_comparison(name, method, rng) for name, method in METHODS.items()]
    metric_names = ("macro_f1", "micro_f1", "far_gt_0_1", "complete_miss")
    flat_p = np.array([value for _, _, _, _, p in results for value in p])
    order = np.argsort(flat_p)
    holm = np.empty_like(flat_p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(flat_p) - rank) * flat_p[idx]))
        holm[idx] = running
    csv_path = ROOT / "paired_hierarchical_bootstrap.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "metric", "replay_minus_baseline_pp", "ci95_low_pp", "ci95_high_pp", "p_two_sided", "p_holm_16", "n_boot", "rng_seed"])
        flat_idx = 0
        for name, point, low, high, p in results:
            for idx, metric in enumerate(metric_names):
                writer.writerow([f"Replay 2k vs {name}", metric, *(100.0 * x for x in (point[idx], low[idx], high[idx])), p[idx], holm[flat_idx], N_BOOT, RNG_SEED])
                flat_idx += 1

    md = [
        "# Paired hierarchical bootstrap",
        "",
        f"Replay 2k is compared with each baseline on identical frozen internal-test pages and three matched seeds. Each of {N_BOOT:,} replicates resamples seeds and then pages within seeds. Values are Replay minus baseline in percentage points; negative values favor Replay for FAR and complete misses. Two-sided bootstrap p-values are descriptive because four methods and four metrics are examined.",
        "",
        "| Baseline | Metric | Delta (pp) | 95% CI (pp) | p | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = ("Macro F1", "Micro F1", "FAR >0.1%", "Complete miss")
    flat_idx = 0
    for name, point, low, high, p in results:
        for idx, label in enumerate(labels):
            md.append(f"| {name} | {label} | {100*point[idx]:+.2f} | [{100*low[idx]:+.2f}, {100*high[idx]:+.2f}] | {p[idx]:.4f} | {holm[flat_idx]:.4f} |")
            flat_idx += 1
    md += [
        "",
        "Interpretation rule: a confidence interval excluding zero is evidence of a paired difference for that metric. This analysis does not authorize a global superiority claim, and multiplicity must be acknowledged.",
        "",
    ]
    (ROOT / "PAIRED_HIERARCHICAL_BOOTSTRAP.md").write_text("\n".join(md), encoding="utf-8")
    print(csv_path)
    print(ROOT / "PAIRED_HIERARCHICAL_BOOTSTRAP.md")


if __name__ == "__main__":
    main()
