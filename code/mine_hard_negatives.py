"""Mine high-confidence false-positive 512x512 crops from strict RTM training data only."""
from __future__ import annotations

import argparse
import os
import csv
import importlib.util
import json
import random
from pathlib import Path

import cv2
import jpegio
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
TRAINER = HERE.parent / "RTM_finetune_DTD" / "train_dtd_rtm_xpu.py"
spec = importlib.util.spec_from_file_location("rtm_trainer", TRAINER)
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split-file", type=Path, default=HERE / "source_aware_split.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=1600)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--keep", type=int, default=800)
    p.add_argument("--seed", type=int, default=20260805)
    return p.parse_args()


def negative_starts(mask, count, rng):
    h, w = mask.shape
    found = []
    for _ in range(count * 40):
        y = tr.aligned_random_start(h, rng)
        x = tr.aligned_random_start(w, rng)
        if not np.any(mask[y:y+tr.TILE, x:x+tr.TILE]):
            found.append((y, x))
            if len(found) == count:
                break
    return found


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.split_file = args.split_file.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    ids = list(split["train"])
    rng = random.Random(args.seed)
    pristine = [x for x in ids if x.startswith("good_")]
    tampered = [x for x in ids if not x.startswith("good_")]
    rng.shuffle(pristine); rng.shuffle(tampered)
    half = args.max_images // 2
    chosen = pristine[:half] + tampered[:args.max_images-len(pristine[:half])]
    rng.shuffle(chosen)

    device = torch.device("xpu:0")
    model, _ = tr.load_model(device, args.checkpoint)
    model.eval()
    rows = []
    with torch.inference_mode():
        for number, image_id in enumerate(chosen, 1):
            image_path = tr.RTM / "JPEGImages" / f"{image_id}.jpg"
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(tr.RTM / "SegmentationClass" / f"{image_id}.png"), 0)
            starts = negative_starts(mask, args.candidates, rng)
            if not starts:
                continue
            jpeg = jpegio.read(str(image_path))
            dct = np.asarray(jpeg.coef_arrays[0])
            q = torch.from_numpy(np.asarray(jpeg.quant_tables[0], dtype=np.int64)).long().unsqueeze(0)
            images, dcts = [], []
            for y, x in starts:
                rgb = cv2.cvtColor(bgr[y:y+tr.TILE, x:x+tr.TILE], cv2.COLOR_BGR2RGB)
                images.append(tr.NORMALIZE(Image.fromarray(rgb)))
                dc = np.clip(np.abs(dct[y:y+tr.TILE, x:x+tr.TILE]), 0, 20).astype(np.int64)
                dcts.append(torch.from_numpy(dc))
            # Batch size 2 is the stable setting for the available Intel Arc XPU.
            prob_chunks = []
            for begin in range(0, len(starts), 2):
                end = min(begin + 2, len(starts))
                logits = model(torch.stack(images[begin:end]).to(device),
                               torch.stack(dcts[begin:end]).to(device),
                               q.repeat(end-begin, 1, 1, 1).to(device))
                prob_chunks.append(torch.softmax(logits, 1)[:, 1].float().cpu().numpy())
            probs = np.concatenate(prob_chunks, axis=0)
            for (y, x), prob in zip(starts, probs):
                flat = prob.reshape(-1)
                k = max(1, flat.size // 100)
                top1 = float(np.partition(flat, -k)[-k:].mean())
                rows.append({"id": image_id, "y": y, "x": x, "score": top1,
                             "area_gt_05": float((prob > .5).mean()),
                             "source": "pristine" if image_id.startswith("good_") else "tampered_background"})
            if number == 1 or number % 100 == 0:
                print(f"mined {number}/{len(chosen)} images, {len(rows)} crops", flush=True)

    rows.sort(key=lambda x: (x["score"], x["area_gt_05"]), reverse=True)
    kept = rows[:args.keep]
    (args.output / "hard_negatives.json").write_text(json.dumps(kept, indent=2), encoding="utf-8")
    with (args.output / "all_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"strict_train_images_considered": len(chosen), "candidate_crops": len(rows),
               "kept": len(kept), "min_kept_score": kept[-1]["score"],
               "mean_kept_score": float(np.mean([x["score"] for x in kept])),
               "pristine_kept": sum(x["source"] == "pristine" for x in kept),
               "tampered_background_kept": sum(x["source"] == "tampered_background" for x in kept)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
