"""Fine-tune official DTD on RTM train split using Intel XPU.

RTM test.txt is never touched. Ten percent of train.txt is held out with a
stratified, deterministic split for patch-level validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import lmdb
import os
import pickle
import random
import six
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import jpegio
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

BASE = Path(os.environ.get("DATA_ROOT", "./data")).resolve()
CODE = BASE / "DocTamper_official_code"
MODELS = CODE / "models"
WEIGHTS = BASE / "DocTamper_pretrained_weights"
RTM = BASE / "RTM_dataset" / "RealTextManipulation"
DEFAULT_OUT = BASE / "测试" / "RTM_finetune_DTD" / "run1"
TILE = 512


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--val-fraction", type=float, default=.10)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-val-steps", type=int, default=0)
    p.add_argument("--positive-crop-prob", type=float, default=.85)
    p.add_argument("--resume", type=Path)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--balanced-mix", action="store_true",
                   help="50% positive, 25% pristine, 25% tampered-image negative patches.")
    p.add_argument("--negative-penalty-weight", type=float, default=0.0)
    p.add_argument("--tamper-class-weight", type=float, default=1.0)
    p.add_argument("--split-file", type=Path)
    p.add_argument("--hard-negative-file", type=Path,
                   help="JSON list produced by hard-negative mining: id/y/x/score.")
    p.add_argument("--hard-negative-prob", type=float, default=.20)
    p.add_argument("--hard-background-weight", type=float, default=0.0,
                   help="Penalty on the highest-probability background pixels in every patch.")
    p.add_argument("--hard-background-fraction", type=float, default=.002,
                   help="Fraction of background pixels selected per patch for hard-pixel loss.")
    p.add_argument("--page-loss-weight", type=float, default=0.0,
                   help="Weight of differentiable tamper-presence supervision derived from shared segmentation logits.")
    p.add_argument("--page-negative-weight", type=float, default=2.0,
                   help="Relative weight for pristine/negative patches in the tamper-presence loss.")
    p.add_argument("--page-top-fraction", type=float, default=.002,
                   help="Fraction of highest-response pixels pooled into the tamper-presence logit.")
    p.add_argument("--far-selection-weight", type=float, default=0.0,
                   help="Checkpoint score penalty per unit pristine area>0.1%% FAR above 20%%.")
    p.add_argument("--train-head-only", action="store_true",
                   help="Freeze feature extractors and adapt only fusion, decoder, and segmentation head.")
    p.add_argument("--doctamper-replay-weight", type=float, default=0.0,
                   help="Weight of supervised DocTamper training-set replay loss.")
    p.add_argument("--doctamper-replay-every", type=int, default=2)
    p.add_argument("--doctamper-replay-samples", type=int, default=2000)
    p.add_argument("--l2sp-weight", type=float, default=0.0,
                   help="L2-SP penalty that anchors trainable parameters to their initial checkpoint values.")
    p.add_argument("--extra-target-batch", action="store_true",
                   help="Backpropagate a second independent RTM batch before each optimizer step (compute/data-matched target-only control).")
    p.add_argument("--lwf-weight", type=float, default=0.0,
                   help="Learning-without-Forgetting distillation weight on target inputs.")
    p.add_argument("--lwf-temperature", type=float, default=2.0)
    p.add_argument("--ewc-weight", type=float, default=0.0,
                   help="Diagonal-Fisher EWC penalty estimated on DocTamper training samples.")
    p.add_argument("--ewc-samples", type=int, default=200,
                   help="Number of deterministic DocTamper training samples used to estimate Fisher information.")
    return p.parse_args()


def category(x):
    return "pristine" if x.startswith("good_") else x.split("_", 1)[0]


def make_split(seed, val_fraction):
    ids = [x.strip() for x in (RTM / "train.txt").read_text().splitlines() if x.strip()]
    groups = defaultdict(list)
    for x in ids: groups[category(x)].append(x)
    rng = random.Random(seed); train, val = [], []
    for _, group in sorted(groups.items()):
        rng.shuffle(group); n = max(1, round(len(group) * val_fraction))
        val.extend(group[:n]); train.extend(group[n:])
    rng.shuffle(train); rng.shuffle(val)
    return train, val


NORMALIZE = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((.485, .455, .406), (.229, .224, .225)),
])


def aligned_random_start(length, rng):
    return rng.randrange(0, (length - TILE) // 8 + 1) * 8


def positive_start(mask, rng):
    ys, xs = np.where(mask != 0)
    if not len(ys): return None
    pick = rng.randrange(len(ys)); py, px = int(ys[pick]), int(xs[pick])
    ylo, yhi = max(0, py - TILE + 1), min(py, mask.shape[0] - TILE)
    xlo, xhi = max(0, px - TILE + 1), min(px, mask.shape[1] - TILE)
    ylo8, yhi8 = (ylo + 7)//8, yhi//8
    xlo8, xhi8 = (xlo + 7)//8, xhi//8
    if ylo8 > yhi8 or xlo8 > xhi8: return None
    return rng.randint(ylo8, yhi8)*8, rng.randint(xlo8, xhi8)*8


class RTMPatchDataset(Dataset):
    def __init__(self, ids, training, seed, positive_crop_prob=.85, balanced_mix=False,
                 hard_negatives=None, hard_negative_prob=.20):
        self.ids, self.training, self.seed = ids, training, seed
        self.positive_crop_prob = positive_crop_prob
        self.balanced_mix = balanced_mix and training
        self.hard_negatives = hard_negatives or []
        self.hard_negative_prob = hard_negative_prob if self.hard_negatives and training else 0.0
        self.pristine_ids = [x for x in ids if x.startswith("good_")]
        self.tampered_ids = [x for x in ids if not x.startswith("good_")]
        self.epoch = 0
    def __len__(self): return len(self.ids)
    def set_epoch(self, epoch): self.epoch = epoch
    def __getitem__(self, index):
        rng = random.Random(self.seed + self.epoch*1_000_003 + index)
        forced_mode = None
        hard_sample = None
        if self.balanced_mix:
            draw = rng.random()
            if draw < self.hard_negative_prob:
                forced_mode = "hard_negative"
                hard_sample = self.hard_negatives[rng.randrange(len(self.hard_negatives))]
                image_id = hard_sample["id"]
            else:
                # Conditional probabilities preserve a 50/25/25 positive/pristine/random-negative
                # mixture within the non-hard replay portion.
                scaled = (draw-self.hard_negative_prob)/(1-self.hard_negative_prob)
                forced_mode = "positive" if scaled < .50 else ("pristine" if scaled < .75 else "tampered_negative")
                pool = self.pristine_ids if forced_mode == "pristine" else self.tampered_ids
                image_id = pool[rng.randrange(len(pool))]
        else:
            image_id = self.ids[index]
        image_path = RTM / "JPEGImages" / f"{image_id}.jpg"
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(RTM / "SegmentationClass" / f"{image_id}.png"), 0)
        jpeg = jpegio.read(str(image_path))
        dct = np.asarray(jpeg.coef_arrays[0])
        qtable = np.asarray(jpeg.quant_tables[0], dtype=np.int64)
        h, w = mask.shape
        if h < TILE or w < TILE: raise ValueError(f"small image {image_id}: {w}x{h}")
        pos = None
        want_positive = mask.max() and (forced_mode == "positive" or
            (forced_mode is None and ((self.training and rng.random() < self.positive_crop_prob) or not self.training)))
        if want_positive:
            pos = positive_start(mask, rng)
        if forced_mode == "hard_negative":
            pos = int(hard_sample["y"]), int(hard_sample["x"])
        elif forced_mode == "tampered_negative":
            best, best_sum = None, None
            for _ in range(32):
                candidate = aligned_random_start(h, rng), aligned_random_start(w, rng)
                cy, cx = candidate; amount = int((mask[cy:cy+TILE, cx:cx+TILE] != 0).sum())
                if best_sum is None or amount < best_sum: best, best_sum = candidate, amount
                if amount == 0: break
            pos = best
        if pos is None:
            pos = aligned_random_start(h, rng), aligned_random_start(w, rng)
        y, x = pos
        rgb = cv2.cvtColor(bgr[y:y+TILE, x:x+TILE], cv2.COLOR_BGR2RGB)
        target = (mask[y:y+TILE, x:x+TILE] != 0).astype(np.int64)
        dc = np.clip(np.abs(dct[y:y+TILE, x:x+TILE]), 0, 20).astype(np.int64)
        return {"image": NORMALIZE(Image.fromarray(rgb)), "target": torch.from_numpy(target),
                "dct": torch.from_numpy(dc), "qtable": torch.from_numpy(qtable).long().unsqueeze(0),
                "id": image_id, "sample_mode": forced_mode or ("validation" if not self.training else "natural")}


class DocTamperReplayDataset(Dataset):
    """Deterministic replay samples from the DocTamper training LMDB only."""
    def __init__(self, count, seed):
        self.root=BASE/"DocTamper_dataset"/"DocTamperV1-TrainingSet"; self.env=None; self.seed=seed
        with lmdb.open(str(self.root),readonly=True,lock=False,readahead=False,meminit=False).begin() as txn:
            total=int(txn.get(b"num-samples"))
        rng=random.Random(seed)
        self.indices=list(range(total)) if count <= 0 else rng.sample(range(total),min(count,total))
        rng.shuffle(self.indices)
        with (CODE/"qt_table.pk").open("rb") as f: self.qtables=pickle.load(f)
    def __len__(self): return len(self.indices)
    def __getitem__(self,pos):
        if self.env is None: self.env=lmdb.open(str(self.root),readonly=True,lock=False,readahead=False,meminit=False)
        index=self.indices[pos]
        with self.env.begin() as txn:
            ib=txn.get(f"image-{index:09d}".encode());lb=txn.get(f"label-{index:09d}".encode())
        im=Image.open(six.BytesIO(ib)).convert("L")
        target=(cv2.imdecode(np.frombuffer(lb,np.uint8),0)!=0).astype(np.int64)
        quality=75+((index*17+self.seed)%26)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            im.save(tmp.name,"JPEG",quality=quality);jpg=jpegio.read(tmp.name)
            dct=jpg.coef_arrays[0].copy();im=Image.open(tmp.name).convert("RGB").copy()
        return {"image":NORMALIZE(im),"target":torch.from_numpy(target),
                "dct":torch.from_numpy(np.clip(np.abs(dct),0,20).astype(np.int64)),
                "qtable":torch.as_tensor(self.qtables[quality]).long().unsqueeze(0)}


def load_model(device, resume=None):
    original = torch.load
    def compat(*a, **kw): kw.setdefault("weights_only", False); return original(*a, **kw)
    torch.load = compat; sys.path.insert(0, str(MODELS)); os.chdir(WEIGHTS)
    import __main__, swins
    for k, v in vars(swins).items():
        if not k.startswith("__"): setattr(__main__, k, v)
    from dtd import seg_dtd
    model = seg_dtd("", 2)
    for m in model.modules():
        if isinstance(m, torch.nn.GELU) and not hasattr(m, "approximate"): m.approximate = "none"
    source = resume or (WEIGHTS / "dtd_doctamper.pth")
    checkpoint = torch.load(source, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device), checkpoint


def batch_counts(pred, target):
    p, t = pred.bool(), target.bool()
    return (int((p&t).sum()), int((p&~t).sum()), int((~p&t).sum()))


def metrics(tp, fp, fn):
    p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.; i=tp/(tp+fp+fn) if tp+fp+fn else 0.
    return p,r,f,i


def segmentation_loss(logits, target, args, dice, ce):
    """Target segmentation objective shared by the main and compute-matched batches."""
    loss=dice(logits,target)+ce(logits,target)
    if args.negative_penalty_weight > 0:
        probs=torch.softmax(logits,dim=1)[:,1]
        negative=(target.flatten(1).sum(1)==0)
        if negative.any():
            flat=probs[negative].flatten(1); k=max(1,flat.shape[1]//100)
            loss=loss+args.negative_penalty_weight*flat.topk(k,dim=1).values.mean()
    if args.hard_background_weight > 0:
        probs=torch.softmax(logits,dim=1)[:,1]
        hard_terms=[]
        for bi in range(target.shape[0]):
            background_probs=probs[bi][target[bi]==0]
            if background_probs.numel():
                k=max(1,int(background_probs.numel()*args.hard_background_fraction))
                chosen=background_probs.topk(k).values.clamp(max=1-1e-6)
                hard_terms.append(-torch.log1p(-chosen).mean())
        if hard_terms:
            loss=loss+args.hard_background_weight*torch.stack(hard_terms).mean()
    if args.page_loss_weight > 0:
        # A shared, parameter-free page/patch head: pool the most suspicious
        # foreground-vs-background logit margins. This sends page-presence
        # gradients through the complete localizer without adding inference
        # parameters or a separately calibrated gate.
        margin=(logits[:,1]-logits[:,0]).flatten(1)
        k=max(1,int(margin.shape[1]*args.page_top_fraction))
        page_logit=margin.topk(k,dim=1).values.mean(dim=1)
        page_target=(target.flatten(1).sum(1)>0).to(page_logit.dtype)
        page_bce=torch.nn.functional.binary_cross_entropy_with_logits(
            page_logit,page_target,reduction="none")
        page_weight=torch.where(page_target>0,torch.ones_like(page_target),
                                torch.full_like(page_target,args.page_negative_weight))
        loss=loss+args.page_loss_weight*(page_bce*page_weight).mean()
    return loss


def estimate_diagonal_fisher(model, parameters, loader, device, ce, max_samples):
    """Estimate empirical diagonal Fisher on source-training examples only."""
    fisher=[torch.zeros_like(p,device=device) for p in parameters]
    reference=[p.detach().clone() for p in parameters]
    model.eval(); seen=0
    for batch in loader:
        model.zero_grad(set_to_none=True)
        image=batch["image"].to(device); target=batch["target"].to(device)
        logits=model(image,batch["dct"].to(device),batch["qtable"].to(device))
        loss=ce(logits,target); loss.backward()
        batch_n=int(image.shape[0])
        for acc,p in zip(fisher,parameters):
            if p.grad is not None: acc.add_(p.grad.detach().square(),alpha=batch_n)
        seen+=batch_n
        if seen>=max_samples: break
    if seen==0: raise RuntimeError("EWC Fisher estimation saw no source samples")
    for acc in fisher: acc.div_(seen)
    model.zero_grad(set_to_none=True)
    return fisher,reference,seen


def main():
    args=args_parser(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.xpu.is_available(): raise RuntimeError("Intel XPU unavailable")
    device=torch.device("xpu:0")
    if args.split_file:
        supplied=json.loads(args.split_file.read_text(encoding="utf-8"))
        train_ids=supplied["train"]
        # New paper protocol uses development for checkpoint selection and keeps
        # internal_test inaccessible to training. Legacy val remains supported
        # only for reproducing exploratory runs.
        val_key="development" if "development" in supplied else "val"
        val_ids=supplied[val_key]
        frozen_ids=set(supplied.get("internal_test", []))
        if frozen_ids & (set(train_ids) | set(val_ids)):
            raise ValueError("Frozen internal_test overlaps train/development")
    else:
        train_ids,val_ids=make_split(args.seed,args.val_fraction);val_key="val";frozen_ids=set()
    split={"seed":args.seed,"train":train_ids,"val":val_ids}
    (args.output/"split.json").write_text(json.dumps(split,indent=2),encoding="utf-8")
    config=vars(args).copy(); config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config["validation_partition"]=val_key
    config["frozen_internal_test_images"]=len(frozen_ids)
    config.update(device=torch.xpu.get_device_name(0),torch=torch.__version__,train_images=len(train_ids),val_images=len(val_ids))
    (args.output/"configuration.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    hard_negatives=[]
    if args.hard_negative_file:
        hard_negatives=json.loads(args.hard_negative_file.read_text(encoding="utf-8"))
        allowed=set(train_ids)
        hard_negatives=[x for x in hard_negatives if x["id"] in allowed]
        if not hard_negatives: raise ValueError("No hard negatives belong to the strict training split")
    train_ds=RTMPatchDataset(train_ids,True,args.seed,args.positive_crop_prob,args.balanced_mix,
                             hard_negatives,args.hard_negative_prob)
    val_ds=RTMPatchDataset(val_ids,False,args.seed,args.positive_crop_prob)
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                            generator=generator,persistent_workers=args.workers>0)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                          persistent_workers=args.workers>0)
    replay_loader=None
    if args.doctamper_replay_weight>0:
        replay_ds=DocTamperReplayDataset(args.doctamper_replay_samples,args.seed)
        replay_loader=DataLoader(replay_ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    ewc_loader=None
    if args.ewc_weight>0:
        ewc_ds=DocTamperReplayDataset(args.ewc_samples,args.seed)
        ewc_loader=DataLoader(ewc_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model,loaded=load_model(device,args.resume)
    teacher=None
    if args.lwf_weight>0:
        # LwF teacher is always the untouched public source checkpoint, never an
        # adapted checkpoint or a source/target evaluation image.
        teacher,_=load_model(device,None)
        teacher.eval()
        for parameter in teacher.parameters(): parameter.requires_grad=False
    if args.train_head_only:
        for parameter in model.parameters(): parameter.requires_grad=False
        for name in ("FU","decoder","segmentation_head"):
            for parameter in getattr(model.model,name).parameters(): parameter.requires_grad=True
    trainable=[parameter for parameter in model.parameters() if parameter.requires_grad]
    l2sp_reference=[parameter.detach().clone() for parameter in trainable] if args.l2sp_weight>0 else None
    config["trainable_parameters"]=sum(parameter.numel() for parameter in trainable)
    config["total_parameters"]=sum(parameter.numel() for parameter in model.parameters())
    (args.output/"configuration.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=args.weight_decay)
    start_epoch=0
    if args.resume:
        start_epoch=int(loaded.get("epoch",-1))+1
        if "optimizer" in loaded and not args.train_head_only:
            optimizer.load_state_dict(loaded["optimizer"])
    from losses import DiceLoss
    dice=DiceLoss(mode="multiclass")
    ce=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,args.tamper_class_weight],device=device),label_smoothing=.001)
    ewc_fisher=ewc_reference=None
    if ewc_loader is not None:
        ewc_fisher,ewc_reference,ewc_seen=estimate_diagonal_fisher(
            model,trainable,ewc_loader,device,ce,args.ewc_samples)
        config["ewc_fisher_samples_seen"]=ewc_seen
        (args.output/"configuration.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    history=[]; best=-1.; log_path=args.output/"history.csv"
    fields=["epoch","train_loss","val_loss","precision","recall","f1","iou",
            "pristine_any_far","pristine_area_gt_01pct_far","pristine_pixel_fpr",
            "train_steps","val_steps","seconds"]
    with log_path.open("a" if start_epoch else "w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        if not start_epoch: writer.writeheader()
        for epoch in range(start_epoch,args.epochs):
            began=time.perf_counter(); train_ds.set_epoch(epoch); model.train(); losses=[]
            replay_iter=iter(replay_loader) if replay_loader is not None else None
            optimizer.zero_grad(set_to_none=True)
            if not args.eval_only:
                train_iter=iter(train_loader);step=0
                while True:
                    try: b=next(train_iter)
                    except StopIteration: break
                    step+=1
                    image=b["image"].to(device); target=b["target"].to(device)
                    dct=b["dct"].to(device); q=b["qtable"].to(device)
                    logits=model(image,dct,q); loss=segmentation_loss(logits,target,args,dice,ce)
                    if teacher is not None:
                        with torch.no_grad(): old_logits=teacher(image,dct,q)
                        temperature=args.lwf_temperature
                        distill=torch.nn.functional.kl_div(
                            torch.log_softmax(logits/temperature,dim=1),
                            torch.softmax(old_logits/temperature,dim=1),reduction="batchmean"
                        )*(temperature**2)/logits[0].numel()
                        loss=loss+args.lwf_weight*distill
                    if l2sp_reference is not None:
                        anchor=torch.stack([(parameter-reference).pow(2).mean()
                                            for parameter,reference in zip(trainable,l2sp_reference)]).mean()
                        loss=loss+args.l2sp_weight*anchor
                    if ewc_fisher is not None:
                        ewc_penalty=torch.stack([
                            (importance*(parameter-reference).square()).sum()
                            for parameter,reference,importance in zip(trainable,ewc_reference,ewc_fisher)
                        ]).sum()
                        loss=loss+args.ewc_weight*ewc_penalty
                    loss.backward(); logged=float(loss.detach().cpu())
                    if args.extra_target_batch:
                        try: eb=next(train_iter)
                        except StopIteration: eb=None
                        if eb is not None:
                            eimage=eb["image"].to(device); etarget=eb["target"].to(device)
                            edct=eb["dct"].to(device); eq=eb["qtable"].to(device)
                            elogits=model(eimage,edct,eq)
                            eloss=segmentation_loss(elogits,etarget,args,dice,ce)
                            if teacher is not None:
                                with torch.no_grad(): eold=teacher(eimage,edct,eq)
                                temperature=args.lwf_temperature
                                edistill=torch.nn.functional.kl_div(
                                    torch.log_softmax(elogits/temperature,dim=1),
                                    torch.softmax(eold/temperature,dim=1),reduction="batchmean"
                                )*(temperature**2)/elogits[0].numel()
                                eloss=eloss+args.lwf_weight*edistill
                            eloss.backward();logged+=float(eloss.detach().cpu())
                    if replay_iter is not None and step%args.doctamper_replay_every==0:
                        try: rb=next(replay_iter)
                        except StopIteration: replay_iter=iter(replay_loader);rb=next(replay_iter)
                        rlogits=model(rb["image"].to(device),rb["dct"].to(device),rb["qtable"].to(device))
                        rtarget=rb["target"].to(device)
                        rloss=args.doctamper_replay_weight*(dice(rlogits,rtarget)+ce(rlogits,rtarget))
                        rloss.backward();logged+=float(rloss.detach().cpu())
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    losses.append(logged)
                    if step==1 or step%25==0: print(f"epoch={epoch} train={step}/{len(train_loader)} loss={np.mean(losses[-25:]):.5f}",flush=True)
                    if args.max_train_steps and step>=args.max_train_steps: break
            model.eval(); vloss=[]; tp=fp=fn=0; clean_any=clean_area01=clean_fp=clean_pixels=clean_images=0
            with torch.inference_mode():
                for vstep,b in enumerate(val_loader,1):
                    image=b["image"].to(device); target=b["target"].to(device)
                    logits=model(image,b["dct"].to(device),b["qtable"].to(device))
                    vloss.append(float((dice(logits,target)+ce(logits,target)).cpu()))
                    pred=logits.argmax(1)
                    for bi in range(target.shape[0]):
                        if b["id"][bi].startswith("good_"):
                            count=int(pred[bi].sum().cpu()); pixels=pred[bi].numel(); clean_images+=1
                            clean_fp+=count;clean_pixels+=pixels;clean_any+=int(count>0);clean_area01+=int(count/pixels>.001)
                        else:
                            a,c,d=batch_counts(pred[bi:bi+1],target[bi:bi+1]); tp+=a;fp+=c;fn+=d
                    if args.max_val_steps and vstep>=args.max_val_steps: break
            p,r,f1,iou=metrics(tp,fp,fn); elapsed=time.perf_counter()-began
            row=dict(epoch=epoch,train_loss=float(np.mean(losses)) if losses else None,val_loss=float(np.mean(vloss)),precision=p,recall=r,f1=f1,iou=iou,
                     pristine_any_far=clean_any/clean_images if clean_images else None,
                     pristine_area_gt_01pct_far=clean_area01/clean_images if clean_images else None,
                     pristine_pixel_fpr=clean_fp/clean_pixels if clean_pixels else None,
                     train_steps=len(losses),val_steps=len(vloss),seconds=elapsed)
            history.append(row);writer.writerow(row);f.flush();print(json.dumps(row),flush=True)
            state={"epoch":epoch,"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"metrics":row,"configuration":config}
            if args.eval_only: break
            torch.save(state,args.output/"checkpoint_latest.pth")
            selection_score=f1-args.far_selection_weight*max(0.0,(clean_area01/clean_images if clean_images else 0.0)-.20)
            state["selection_score"]=selection_score
            torch.save(state,args.output/f"checkpoint_epoch_{epoch}.pth")
            if selection_score>best: best=selection_score;torch.save(state,args.output/"checkpoint_best.pth")
    (args.output/"COMPLETED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"),encoding="utf-8")


if __name__=="__main__": main()
