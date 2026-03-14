"""
train_kaggle.py v2 — Fixed version.
Key fixes:
  1. DataParallel bug fixed — compute_loss called on base_model
  2. RGB images (3-channel)
  3. ONNX export works correctly
  4. Better per-class metrics logging
  5. Consolidated setup — no session restart issues
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

warnings.filterwarnings("ignore")

sys.path.insert(0, '/kaggle/working')
sys.path.insert(0, '/kaggle/working/model')

from floodnet import FloodNet, export_to_onnx
from preprocess import build_dataloaders, prepare_floodnet_csv, generate_synthetic_csv, DAMAGE_LABELS


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",   default="")
    p.add_argument("--val_csv",     default="")
    p.add_argument("--output_dir",  default="/kaggle/working/output")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--fp16",        action="store_true", default=True)
    p.add_argument("--use_text",    action="store_true", default=True)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--grad_accum",  type=int,   default=2)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
class Metrics:
    def __init__(self, device):
        self.acc = MulticlassAccuracy(num_classes=4).to(device)
        self.f1  = MulticlassF1Score(num_classes=4, average="macro").to(device)
        self.losses = []

    def update(self, logits, labels, loss):
        preds = logits.argmax(dim=-1)
        self.acc.update(preds, labels)
        self.f1.update(preds, labels)
        self.losses.append(loss)

    def compute(self):
        return {
            "loss": float(np.mean(self.losses)),
            "acc":  self.acc.compute().item(),
            "f1":   self.f1.compute().item(),
        }

    def reset(self):
        self.acc.reset(); self.f1.reset(); self.losses.clear()


# ─────────────────────────────────────────────
# TRAIN EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, base_model, loader, optimizer, scheduler, scaler, device, fp16, grad_accum, epoch, metrics):
    model.train()
    metrics.reset()
    optimizer.zero_grad(set_to_none=True)

    # FIX: criterion defined here — no DataParallel issues
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    t0 = time.time()
    for step, batch in enumerate(loader):
        image   = batch["image"].to(device, non_blocking=True)
        weather = batch["weather"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)
        input_ids  = batch.get("input_ids")
        attn_mask  = batch.get("attention_mask")
        if input_ids is not None:
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

        with autocast(enabled=fp16):
            out  = model(image, weather, input_ids, attn_mask)
            # FIX: use criterion directly, not model.compute_loss
            loss = criterion(out["logits"], labels) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metrics.update(out["logits"].detach(), labels, loss.item() * grad_accum)

        if step % 5 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (step + 1)) * (len(loader) - step)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [Epoch {epoch}] Step {step:3d}/{len(loader)} | "
                  f"Loss: {loss.item()*grad_accum:.4f} | LR: {lr_now:.2e} | ETA: {eta:.0f}s", end="\r")

    print()
    return metrics.compute()


# ─────────────────────────────────────────────
# VAL EPOCH
# ─────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, device, fp16, metrics):
    model.eval()
    metrics.reset()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    all_preds, all_labels = [], []

    for batch in loader:
        image   = batch["image"].to(device, non_blocking=True)
        weather = batch["weather"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)
        input_ids = batch.get("input_ids")
        attn_mask = batch.get("attention_mask")
        if input_ids is not None:
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

        with autocast(enabled=fp16):
            out  = model(image, weather, input_ids, attn_mask)
            loss = criterion(out["logits"], labels)

        metrics.update(out["logits"], labels, loss.item())
        all_preds.extend(out["logits"].argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    result = metrics.compute()

    # Per-class accuracy
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    for i, name in enumerate(DAMAGE_LABELS):
        mask = all_labels == i
        if mask.sum() > 0:
            result[f"acc_{i}"] = float((all_preds[mask] == i).mean())
        else:
            result[f"acc_{i}"] = 0.0

    return result


# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def train(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[GPU] {n_gpus} GPU(s) available")

    # Data
    train_csv = args.train_csv
    val_csv   = args.val_csv

    if not train_csv or not Path(train_csv).exists():
        print("[Data] No CSV provided — using synthetic data")
        train_csv, val_csv = generate_synthetic_csv(
            str(output_dir / "data"), n=2000
        )

    train_loader, val_loader = build_dataloaders(
        train_csv, val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_text=args.use_text,
    )

    # Model — FIX: RGB (3 channels)
    print(f"[Model] Building FloodNet v2 (RGB, use_text={args.use_text})")
    model = FloodNet(
        sar_channels=3,          # FIX: RGB
        weather_dim=10,
        use_text=args.use_text,
        pretrained_image=True,
        dropout=0.3,
    )

    # FIX: get base_model reference BEFORE DataParallel
    base_model = model

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"[Model] DataParallel on {n_gpus} GPUs")

    model = model.to(device)

    total = sum(p.numel() for p in base_model.parameters()) / 1e6
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1e6
    print(f"[Model] Total: {total:.1f}M | Trainable: {trainable:.1f}M")

    # Optimizer — differential LR
    param_groups = [
        {"params": base_model.classifier.parameters(),  "lr": args.lr},
        {"params": base_model.fusion.parameters(),      "lr": args.lr},
        {"params": base_model.weather_enc.parameters(), "lr": args.lr},
        {"params": base_model.image_enc.parameters(),   "lr": args.lr * 0.1},
    ]
    if args.use_text and base_model.text_enc is not None:
        param_groups.append({"params": base_model.text_enc.parameters(), "lr": 1e-5})

    optimizer = AdamW(param_groups, weight_decay=1e-4, eps=1e-6)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    max_lrs = [args.lr, args.lr, args.lr, args.lr * 0.1]
    if args.use_text:
        max_lrs.append(1e-5)

    scheduler = OneCycleLR(
        optimizer, max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = GradScaler(enabled=args.fp16)

    # Train
    train_metrics = Metrics(device)
    val_metrics   = Metrics(device)
    history = []
    best_f1 = 0.0
    best_ckpt = output_dir / "best_model.pth"

    print(f"\n{'='*55}")
    print(f"  FloodNet v2 | {args.epochs} epochs | batch={args.batch_size} | FP16={args.fp16}")
    print(f"{'='*55}\n")

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        print(f"── Epoch {epoch}/{args.epochs} ──────────────────────────")

        tr = train_epoch(model, base_model, train_loader, optimizer, scheduler,
                         scaler, device, args.fp16, args.grad_accum, epoch, train_metrics)
        vl = val_epoch(model, val_loader, device, args.fp16, val_metrics)

        elapsed = time.time() - t_epoch

        print(f"  TRAIN  loss={tr['loss']:.4f}  acc={tr['acc']:.4f}  f1={tr['f1']:.4f}")
        print(f"  VAL    loss={vl['loss']:.4f}  acc={vl['acc']:.4f}  f1={vl['f1']:.4f}  [{elapsed:.0f}s]")

        for i, name in enumerate(DAMAGE_LABELS):
            print(f"    {name:>20}: {vl.get(f'acc_{i}', 0):.4f}")

        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in vl.items()}})

        if vl["f1"] > best_f1:
            best_f1 = vl["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "best_f1": best_f1,
                "config": vars(args),
                "sar_channels": 3,  # save this for ONNX export
            }, best_ckpt)
            print(f"  ✓ Best model saved (F1={best_f1:.4f})")

    total_time = (time.time() - t_start) / 60
    print(f"\n{'='*55}")
    print(f"  Done in {total_time:.1f} min | Best F1: {best_f1:.4f}")
    print(f"{'='*55}")

    # Save history
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    # ONNX Export — FIX: load checkpoint into correct architecture
    print("\n[ONNX] Exporting …")
    ckpt = torch.load(best_ckpt, map_location="cpu")

    export_model = FloodNet(
        sar_channels=3,             # FIX: must match training
        weather_dim=10,
        use_text=args.use_text,     # FIX: must match training
        pretrained_image=False,
    )
    export_model.load_state_dict(ckpt["model_state_dict"], strict=True)

    onnx_path = str(output_dir / "floodnet.onnx")
    export_to_onnx(export_model, onnx_path, device="cpu")

    # Model card
    with open(output_dir / "model_card.json", "w") as f:
        json.dump({
            "model": "FloodNet v2",
            "best_val_f1": best_f1,
            "epochs": args.epochs,
            "sar_channels": 3,
            "use_text": args.use_text,
            "damage_classes": DAMAGE_LABELS,
            "training_time_min": total_time,
        }, f, indent=2)

    print(f"\n[Output] Files saved to: {output_dir}")
    for f in output_dir.iterdir():
        mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {mb:.1f} MB")

    return str(best_ckpt), onnx_path


if __name__ == "__main__":
    args = parse_args()
    train(args)
