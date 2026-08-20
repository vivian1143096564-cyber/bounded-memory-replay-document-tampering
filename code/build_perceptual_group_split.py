"""Build a leakage-controlled RTM train/development/internal-test split.

Only RTM train.txt is read.  Images connected by a 64-bit difference-hash
distance <= ``--max-hamming`` are kept in one partition.  This is a pragmatic
fallback when the distributed dataset has no document-source metadata; the
result must be described as perceptual-group-disjoint, not source-disjoint.
"""

from __future__ import annotations

import argparse
import os
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rtm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--max-hamming", type=int, default=4)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--development-fraction", type=float, default=0.15)
    return p.parse_args()


def category(image_id: str) -> str:
    return "pristine" if image_id.startswith("good_") else image_id.split("_", 1)[0]


def dhash64(path: Path) -> int:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def main() -> None:
    args = parse_args()
    internal_fraction = 1.0 - args.train_fraction - args.development_fraction
    if min(args.train_fraction, args.development_fraction, internal_fraction) <= 0:
        raise ValueError("All three split fractions must be positive")
    ids = [x.strip() for x in (args.rtm / "train.txt").read_text().splitlines() if x.strip()]
    image_dir = args.rtm / "JPEGImages"
    hashes = [dhash64(image_dir / f"{image_id}.jpg") for image_id in ids]

    uf = UnionFind(len(ids))
    # Exhaustive comparison is intentionally used once during protocol creation.
    # 5.8k images produce about 16.8M inexpensive integer comparisons.
    for i, h in enumerate(hashes):
        for j in range(i):
            if (h ^ hashes[j]).bit_count() <= args.max_hamming:
                uf.union(i, j)

    components = defaultdict(list)
    for i, image_id in enumerate(ids):
        components[uf.find(i)].append(image_id)
    groups = list(components.values())
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    partitions = {"train": [], "development": [], "internal_test": []}
    group_ids = {name: [] for name in partitions}
    fractions = {
        "train": args.train_fraction,
        "development": args.development_fraction,
        "internal_test": internal_fraction,
    }
    total_by_cat = Counter(category(x) for x in ids)
    target_total = {name: len(ids) * frac for name, frac in fractions.items()}
    target_cat = {
        name: {cat: count * fractions[name] for cat, count in total_by_cat.items()}
        for name in partitions
    }
    assigned_cat = {name: Counter() for name in partitions}

    for group_index, group in enumerate(groups):
        group_cat = Counter(category(x) for x in group)
        best_name, best_score = None, None
        for name in partitions:
            total_after = len(partitions[name]) + len(group)
            score = abs(total_after - target_total[name]) / max(1.0, target_total[name])
            for cat, target in target_cat[name].items():
                after = assigned_cat[name][cat] + group_cat[cat]
                score += abs(after - target) / max(1.0, target)
            # Strongly discourage exceeding the intended size while another split is deficient.
            if total_after > target_total[name] * 1.03:
                score += 5.0 * (total_after / target_total[name] - 1.03)
            if best_score is None or score < best_score:
                best_name, best_score = name, score
        partitions[best_name].extend(group)
        assigned_cat[best_name].update(group_cat)
        group_ids[best_name].append(group_index)

    # The greedy objective above minimizes the wrong absolute state early on if
    # evaluated independently. Repair by deterministic deficit filling instead.
    # Reassign from scratch, always choosing the partition with the largest
    # normalized deficit for the current group's categories and total size.
    partitions = {"train": [], "development": [], "internal_test": []}
    group_ids = {name: [] for name in partitions}
    assigned_cat = {name: Counter() for name in partitions}
    for group_index, group in enumerate(groups):
        group_cat = Counter(category(x) for x in group)
        scores = {}
        for name in partitions:
            total_deficit = (target_total[name] - len(partitions[name])) / max(1.0, target_total[name])
            cat_deficit = sum(
                group_cat[cat] * (target_cat[name][cat] - assigned_cat[name][cat]) / max(1.0, target_cat[name][cat])
                for cat in group_cat
            ) / max(1, len(group))
            scores[name] = total_deficit + cat_deficit
        chosen = max(scores, key=lambda name: (scores[name], fractions[name]))
        partitions[chosen].extend(group)
        assigned_cat[chosen].update(group_cat)
        group_ids[chosen].append(group_index)

    membership = {}
    for name, members in partitions.items():
        for image_id in members:
            membership[image_id] = name
    cross_group_violations = 0
    for group in groups:
        if len({membership[x] for x in group}) != 1:
            cross_group_violations += 1

    args.output.mkdir(parents=True, exist_ok=True)
    split = {
        "protocol": "RTM train.txt only; 64-bit dHash connected components with Hamming distance <= %d; group-disjoint 70/15/15 allocation." % args.max_hamming,
        "seed": args.seed,
        "max_hamming": args.max_hamming,
        "train": sorted(partitions["train"]),
        "development": sorted(partitions["development"]),
        "internal_test": sorted(partitions["internal_test"]),
    }
    (args.output / "perceptual_group_split.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    group_manifest = [
        {"group_id": i, "size": len(group), "members": sorted(group), "categories": dict(Counter(category(x) for x in group))}
        for i, group in enumerate(groups)
    ]
    (args.output / "perceptual_groups.json").write_text(
        json.dumps(group_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "dataset_images_from_train_txt": len(ids),
        "groups": len(groups),
        "multi_image_groups": sum(len(x) > 1 for x in groups),
        "largest_group": max(map(len, groups)),
        "cross_group_violations": cross_group_violations,
        "partition_counts": {name: len(members) for name, members in partitions.items()},
        "partition_category_counts": {name: dict(Counter(category(x) for x in members)) for name, members in partitions.items()},
        "official_test_read": False,
        "naming_warning": "This split is perceptual-group-disjoint, not verified document-source-disjoint.",
    }
    (args.output / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
