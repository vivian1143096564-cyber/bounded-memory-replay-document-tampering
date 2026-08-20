from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import cv2
import jpegio
import lmdb
import numpy as np
import six
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--minq", type=int, default=75)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


class DocTamperDataset(Dataset):
    def __init__(self, code_root: Path, data_root: Path, name: str, minq: int, limit: int):
        self.root = data_root / name
        self.env = None
        with lmdb.open(str(self.root), readonly=True, lock=False, readahead=False, meminit=False).begin() as txn:
            total = int(txn.get(b"num-samples"))
        self.count = min(total, limit) if limit else total
        with (code_root / "qt_table.pk").open("rb") as f:
            self.qtables = pickle.load(f)
        with (code_root / "pks" / f"{name}_{minq}.pk").open("rb") as f:
            self.records = pickle.load(f)
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.455, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.env is None:
            self.env = lmdb.open(str(self.root), readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            ib = txn.get(f"image-{index:09d}".encode())
            lb = txn.get(f"label-{index:09d}".encode())
        image = Image.open(six.BytesIO(ib)).convert("L")
        target = (cv2.imdecode(np.frombuffer(lb, np.uint8), cv2.IMREAD_GRAYSCALE) != 0).astype(np.uint8)
        qualities = [int(x) for x in self.records[index]]
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            for quality in qualities:
                image.save(tmp.name, "JPEG", quality=quality)
                image = Image.open(tmp.name).copy()
            dct = jpegio.read(tmp.name).coef_arrays[0].copy()
        image = image.convert("RGB")
        return (
            self.normalize(image),
            torch.from_numpy(np.clip(np.abs(dct), 0, 20)),
            torch.as_tensor(self.qtables[qualities[-1]], dtype=torch.long).unsqueeze(0),
            torch.from_numpy(target).long(),
        )


def load_model(args, device):
    original_load = torch.load
    def compatible_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return original_load(*a, **kw)
    torch.load = compatible_load
    sys.path.insert(0, str(args.code_root / "models"))
    os.chdir(args.weights_root)
    import __main__, swins
    for key, value in vars(swins).items():
        if not key.startswith("__"):
            setattr(__main__, key, value)
    from dtd import seg_dtd
    model = seg_dtd("", 2)
    for module in model.modules():
        if isinstance(module, torch.nn.GELU) and not hasattr(module, "approximate"):
            module.approximate = "none"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return model.eval().to(device)


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU unavailable")
    device = torch.device("xpu:0")
    dataset = DocTamperDataset(args.code_root, args.data_root, args.dataset, args.minq, args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    model = load_model(args, device)
    totals = np.zeros(3, dtype=np.int64)
    macro = np.zeros(4, dtype=np.float64)
    model_seconds = 0.0
    began = time.perf_counter()
    seen = 0
    with torch.inference_mode():
        for image, dct, qtable, target in loader:
            image, dct, qtable = image.to(device), dct.to(device), qtable.to(device)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            pred = model(image, dct, qtable).argmax(1).cpu().numpy().astype(bool)
            torch.xpu.synchronize()
            model_seconds += time.perf_counter() - t0
            truth = target.numpy().astype(bool)
            for p, t in zip(pred, truth):
                tp = int(np.logical_and(p, t).sum())
                fp = int(np.logical_and(p, ~t).sum())
                fn = int(np.logical_and(~p, t).sum())
                totals += (tp, fp, fn)
                s = scores(tp, fp, fn)
                macro += [s["precision"], s["recall"], s["f1"], s["iou"]]
                seen += 1
            if seen % 100 == 0 or seen == len(dataset):
                print(f"processed={seen}/{len(dataset)}", flush=True)
    wall = time.perf_counter() - began
    micro = scores(*[int(x) for x in totals])
    report = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "samples": seen,
        "macro": dict(zip(("precision", "recall", "f1", "iou"), (macro / seen).tolist())),
        "micro": micro,
        "model_seconds_per_image": model_seconds / seen,
        "wall_seconds_per_image": wall / seen,
        "total_wall_seconds": wall,
        "device": torch.xpu.get_device_name(0),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
