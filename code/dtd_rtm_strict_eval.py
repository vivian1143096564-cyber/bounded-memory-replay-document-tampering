"""Zero-shot evaluation of the official DTD/DocTamper checkpoint on RTM.

This follows the official high-resolution inference protocol: crop to an
8-pixel boundary, use 512x512 tiles (including right/bottom boundary tiles),
and retain the JPEG's original luminance DCT coefficients and quantization
table.  RTM is never used for training or threshold tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import jpegio
import numpy as np
import torch
import torchvision
import joblib
from PIL import Image


BASE = Path(os.environ.get("DATA_ROOT", "./data")).resolve()
CODE_ROOT = BASE / "DocTamper_official_code"
MODEL_ROOT = CODE_ROOT / "models"
WEIGHT_ROOT = BASE / "DocTamper_pretrained_weights"
RTM_ROOT = BASE / "RTM_dataset" / "RealTextManipulation"
DEFAULT_OUTPUT = BASE / "测试" / "RTM_zero_shot_DTD" / "full_results"
TILE = 512
THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--balanced-smoke", type=int, default=0,
                        help="Select half tampered and half pristine samples.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted CSV run.")
    parser.add_argument("--checkpoint", type=Path, default=WEIGHT_ROOT / "dtd_doctamper.pth")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--gate-model", type=Path)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--split-file", type=Path,
                        help="Required JSON split file from protocol/perceptual_group_split.json.")
    parser.add_argument("--split-key", choices=["val", "development", "internal_test"], default=None,
                        help="Partition to evaluate from --split-file. Required for three-way paper splits.")
    return parser.parse_args()


def manipulation_type(name: str) -> str:
    return "pristine" if name.startswith("good_") else name.split("_", 1)[0]


def select_ids(args) -> list[str]:
    if args.split_file:
        supplied = json.loads(args.split_file.read_text(encoding="utf-8"))
        if "internal_test" in supplied and args.split_key is None:
            raise ValueError("--split-key is required for a three-way split; choose development or internal_test explicitly")
        split_key = args.split_key or "val"
        ids = supplied[split_key]
    else:
        raise ValueError("--split-file is required; official RTM test access is disabled in this bundle.")
    if args.balanced_smoke:
        rng = random.Random(args.seed)
        clean = [x for x in ids if x.startswith("good_")]
        forged = [x for x in ids if not x.startswith("good_")]
        n_clean = args.balanced_smoke // 2
        n_forged = args.balanced_smoke - n_clean
        ids = rng.sample(clean, min(n_clean, len(clean))) + rng.sample(forged, min(n_forged, len(forged)))
        rng.shuffle(ids)
    elif args.limit:
        ids = ids[:args.limit]
    return ids


def tile_positions(height: int, width: int) -> list[tuple[int, int]]:
    """Match crop_img ordering from the official RTM inference script."""
    hs, ws = height // TILE, width // TILE
    pos = [(i * TILE, j * TILE) for i in range(hs) for j in range(ws)]
    if width % TILE:
        pos += [(i * TILE, width - TILE) for i in range(hs)]
    if height % TILE:
        pos += [(height - TILE, j * TILE) for j in range(ws)]
    if height % TILE and width % TILE:
        pos.append((height - TILE, width - TILE))
    return pos


def load_model(device, checkpoint_path):
    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load
    sys.path.insert(0, str(MODEL_ROOT))
    os.chdir(WEIGHT_ROOT)
    import __main__
    import swins
    for name, value in vars(swins).items():
        if not name.startswith("__"):
            setattr(__main__, name, value)
    from dtd import seg_dtd
    model = seg_dtd("", 2)
    for module in model.modules():
        if isinstance(module, torch.nn.GELU) and not hasattr(module, "approximate"):
            module.approximate = "none"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = {k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


NORMALIZE = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.455, 0.406), (0.229, 0.224, 0.225)),
])


def infer_one(model, device, image_path: Path, batch_size: int, gate=None, gate_threshold=.5):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read {image_path}")
    original_h, original_w = bgr.shape[:2]
    h, w = (original_h // 8) * 8, (original_w // 8) * 8
    if h < TILE or w < TILE:
        raise ValueError(f"Image smaller than {TILE}: {original_w}x{original_h}")
    bgr8 = bgr[:h, :w]
    jpeg = jpegio.read(str(image_path))
    dct = np.asarray(jpeg.coef_arrays[0])[:h, :w]
    qtable = np.asarray(jpeg.quant_tables[0], dtype=np.int64)
    positions = tile_positions(h, w)
    prob = np.zeros((h, w), np.float32)
    model_seconds = 0.0
    q_base = torch.from_numpy(qtable).long().unsqueeze(0).unsqueeze(0)

    for start in range(0, len(positions), batch_size):
        batch_pos = positions[start:start + batch_size]
        rgb_tensors, dct_tensors = [], []
        for y, x in batch_pos:
            rgb = cv2.cvtColor(bgr8[y:y+TILE, x:x+TILE], cv2.COLOR_BGR2RGB)
            rgb_tensors.append(NORMALIZE(Image.fromarray(rgb)))
            dc = np.clip(np.abs(dct[y:y+TILE, x:x+TILE]), 0, 20).astype(np.int64)
            dct_tensors.append(torch.from_numpy(dc))
        image_t = torch.stack(rgb_tensors).to(device)
        dct_t = torch.stack(dct_tensors).to(device)
        q_t = q_base.repeat(len(batch_pos), 1, 1, 1).to(device)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(image_t, dct_t, q_t)
            batch_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        torch.xpu.synchronize()
        model_seconds += time.perf_counter() - t0
        gate_scores = (gate.predict_proba(np.asarray([probability_features(p) for p in batch_prob], np.float32))[:,1]
                       if gate is not None else np.ones(len(batch_prob)))
        for (y, x), tile_prob, gate_score in zip(batch_pos, batch_prob, gate_scores):
            if gate_score < gate_threshold:
                tile_prob = np.zeros_like(tile_prob)
            prob[y:y+TILE, x:x+TILE] = tile_prob

    full_prob = np.zeros((original_h, original_w), np.float32)
    full_prob[:h, :w] = prob
    return bgr, full_prob, len(positions), model_seconds


def probability_features(prob):
    a=np.asarray(prob,np.float32);flat=a.ravel();out=[float(flat.mean()),float(flat.std())]
    out+=np.quantile(flat,[.5,.75,.9,.95,.99,.995,.999,1]).tolist()
    out+=[float((flat>=t).mean()) for t in (.1,.2,.3,.4,.5,.6,.7,.8,.9)]
    ordered=np.sort(flat);out+=[float(ordered[-max(1,int(len(flat)*f)):].mean()) for f in (.0001,.001,.01,.05)]
    for gy in range(2):
        for gx in range(2):
            q=a[gy*256:(gy+1)*256,gx*256:(gx+1)*256];out.extend([float(q.mean()),float(q.max())])
    binary=(a>=.5).astype(np.uint8);n,_,stats,_=cv2.connectedComponentsWithStats(binary,8)
    areas=stats[1:,cv2.CC_STAT_AREA] if n>1 else np.array([],dtype=int)
    out.extend([float(len(areas)),float(areas.max()/a.size) if len(areas) else 0.,float((areas>=8).sum())])
    return out


def counts(pred: np.ndarray, target: np.ndarray):
    p, t = pred.astype(bool), target.astype(bool)
    return (int(np.logical_and(p, t).sum()), int(np.logical_and(p, ~t).sum()),
            int(np.logical_and(~p, t).sum()), int(np.logical_and(~p, ~t).sum()))


def safe_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return precision, recall, f1, iou


FIELDS = ["image_id", "tampering_type", "is_tampered", "width", "height", "tiles",
          "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou",
          "gt_positive_pixels", "pred_positive_pixels", "pred_area_ratio",
          "model_seconds", "total_seconds", "status", "error"]


def bootstrap_ci(values, seed=20260804, rounds=10000):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(rounds)
    for i in range(rounds):
        means[i] = rng.choice(a, len(a), replace=True).mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(rows, args):
    ok = [r for r in rows if r["status"] == "ok"]
    # Rows loaded during --resume come from CSV and therefore contain strings,
    # while newly inferred rows contain integers. Normalize at comparison time
    # so resumed and uninterrupted evaluations summarize identically.
    tampered = [r for r in ok if int(r["is_tampered"]) == 1]
    pristine = [r for r in ok if int(r["is_tampered"]) == 0]
    totals = {k: sum(int(r[k]) for r in tampered) for k in ("tp", "fp", "fn", "tn")}
    micro = safe_metrics(totals["tp"], totals["fp"], totals["fn"])
    macro = {}
    for key in ("precision", "recall", "f1", "iou"):
        vals = [float(r[key]) for r in tampered]
        macro[key] = {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                      "std": float(np.std(vals)), "bootstrap_95_ci": bootstrap_ci(vals)}
    total_clean_pixels = sum(int(r["fp"]) + int(r["tn"]) for r in pristine)
    total_clean_fp = sum(int(r["fp"]) for r in pristine)
    area = np.array([float(r["pred_area_ratio"]) for r in pristine])
    false_alarm_rates = {str(th): float(np.mean(area > th)) if len(area) else None
                         for th in (0.0, 0.0001, 0.001, 0.01)}
    report = {
        "protocol": (
            f"DTD checkpoint evaluation on split:{args.split_key or 'val'}, official 512px tiling"
            "DTD checkpoint evaluation on the declared split; official RTM test access disabled"
        ),
        "counts": {"evaluated": len(ok), "failed": len(rows)-len(ok),
                   "tampered": len(tampered), "pristine": len(pristine)},
        "tampered_localization": {
            "pixel_micro": dict(zip(("precision", "recall", "f1", "iou"), micro)),
            "image_macro": macro,
            "complete_miss_rate": float(np.mean([int(r["pred_positive_pixels"]) == 0 for r in tampered])),
            "recall_below_0.1_rate": float(np.mean([float(r["recall"]) < .1 for r in tampered])),
            "recall_below_0.5_rate": float(np.mean([float(r["recall"]) < .5 for r in tampered])),
        },
        "pristine_false_alarm": {
            "pixel_fpr": total_clean_fp / total_clean_pixels if total_clean_pixels else None,
            "image_false_alarm_rate_by_predicted_area": false_alarm_rates,
            "predicted_area_ratio_mean": float(area.mean()) if len(area) else None,
            "predicted_area_ratio_median": float(np.median(area)) if len(area) else None,
            "predicted_area_ratio_p95": float(np.quantile(area, .95)) if len(area) else None,
        },
        "speed": {
            "total_model_seconds": sum(float(r["model_seconds"]) for r in ok),
            "mean_model_seconds_per_image": float(np.mean([float(r["model_seconds"]) for r in ok])),
            "mean_total_seconds_per_image": float(np.mean([float(r["total_seconds"]) for r in ok])),
            "total_tiles": sum(int(r["tiles"]) for r in ok),
        },
    }
    return report


def write_type_summary(rows, path):
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[r["tampering_type"]].append(r)
    fields = ["type", "images", "macro_precision", "macro_recall", "macro_f1", "macro_iou",
              "micro_precision", "micro_recall", "micro_f1", "micro_iou", "complete_misses"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name, group in sorted(groups.items()):
            tp, fp, fn = (sum(int(r[k]) for r in group) for k in ("tp", "fp", "fn"))
            mic = safe_metrics(tp, fp, fn)
            writer.writerow({"type": name, "images": len(group),
                "macro_precision": np.mean([float(r["precision"]) for r in group]),
                "macro_recall": np.mean([float(r["recall"]) for r in group]),
                "macro_f1": np.mean([float(r["f1"]) for r in group]),
                "macro_iou": np.mean([float(r["iou"]) for r in group]),
                "micro_precision": mic[0], "micro_recall": mic[1], "micro_f1": mic[2], "micro_iou": mic[3],
                "complete_misses": sum(int(r["pred_positive_pixels"]) == 0 for r in group)})


def main():
    args = parse_args()
    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output / "predictions"; pred_dir.mkdir(exist_ok=True)
    ids = select_ids(args)
    config = {"checkpoint": str(args.checkpoint), "dataset": str(RTM_ROOT),
              "split": (args.split_key or "val") if args.split_file else "test", "images_requested": len(ids), "threshold": args.threshold,
              "tile_size": TILE, "batch_size": args.batch_size, "seed": args.seed,
              "gate_model": str(args.gate_model) if args.gate_model else None,"gate_threshold":args.gate_threshold,
              "device": torch.xpu.get_device_name(0), "torch": torch.__version__}
    (args.output / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "RUNNING").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    model = load_model(torch.device("xpu:0"), args.checkpoint)
    gate = joblib.load(args.gate_model) if args.gate_model else None
    rows = []
    csv_path = args.output / "sample_metrics.csv"
    if args.resume and csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as existing:
            rows = list(csv.DictReader(existing))
        done = {r["image_id"] for r in rows if r["status"] == "ok"}
        ids = [x for x in ids if x not in done]
        print(f"Resuming after {len(done)} completed images; {len(ids)} remain", flush=True)
    mode = "a" if args.resume and csv_path.exists() else "w"
    with csv_path.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        already = len(rows)
        for local_index, image_id in enumerate(ids, 1):
            index = already + local_index
            started = time.perf_counter()
            row = {k: "" for k in FIELDS}
            row.update(image_id=image_id, tampering_type=manipulation_type(image_id),
                       is_tampered=int(not image_id.startswith("good_")), status="failed")
            try:
                image_path = RTM_ROOT / "JPEGImages" / f"{image_id}.jpg"
                mask_path = RTM_ROOT / "SegmentationClass" / f"{image_id}.png"
                bgr, prob, tiles, model_seconds = infer_one(model, torch.device("xpu:0"), image_path, args.batch_size, gate, args.gate_threshold)
                target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if target is None or target.shape != prob.shape:
                    raise ValueError(f"mask shape mismatch: {None if target is None else target.shape} vs {prob.shape}")
                pred = prob >= args.threshold
                tp, fp, fn, tn = counts(pred, target != 0)
                precision, recall, f1, iou = safe_metrics(tp, fp, fn)
                h, w = target.shape
                row.update(width=w, height=h, tiles=tiles, tp=tp, fp=fp, fn=fn, tn=tn,
                           precision=precision, recall=recall, f1=f1, iou=iou,
                           gt_positive_pixels=int((target != 0).sum()), pred_positive_pixels=int(pred.sum()),
                           pred_area_ratio=float(pred.mean()), model_seconds=model_seconds,
                           status="ok", error="")
                if not args.no_predictions:
                    cv2.imwrite(str(pred_dir / f"{image_id}.png"), np.rint(prob * 255).astype(np.uint8))
            except Exception as exc:
                row["error"] = repr(exc)
            row["total_seconds"] = time.perf_counter() - started
            rows.append(row); writer.writerow(row); f.flush()
            if index == 1 or index % 10 == 0 or local_index == len(ids):
                elapsed = sum(float(r["total_seconds"]) for r in rows)
                print(f"[{index}/{already + len(ids)}] {image_id} {row['status']} mean={elapsed/index:.2f}s", flush=True)

    report = summarize(rows, args)
    (args.output / "overall_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_type_summary(rows, args.output / "type_summary.csv")
    misses = sorted((r for r in rows if r["status"] == "ok" and r["is_tampered"] == 1), key=lambda r: float(r["recall"]))
    false_alarms = sorted((r for r in rows if r["status"] == "ok" and r["is_tampered"] == 0), key=lambda r: float(r["pred_area_ratio"]), reverse=True)
    for name, data in (("missed_cases.csv", misses), ("pristine_false_alarms.csv", false_alarms)):
        with (args.output / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(data)
    (args.output / "RUNNING").unlink(missing_ok=True)
    (args.output / "COMPLETED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
