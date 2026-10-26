import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import os
import random
import gc
import json
import time

from prepare_carla import (
    CARLA_RUNS_DIR,
    MODEL_SAVE_DIR,
    CACHE_DIR,
    count_shards,
    get_shard_path,
    SHARD_SIZE,
)

try:
    from prepare_dsec import DSEC_CACHE_DIR
except ImportError:
    DSEC_CACHE_DIR = os.path.join(os.path.dirname(CACHE_DIR), "dsec_cache")


SPLIT_SEED = 1234


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EventFlowDataset(Dataset):
    def __init__(self, index, shuffle_shards=True, max_cache_shards=20):
        self.samples = []

        shard_groups = {}
        for shard_path, n in index:
            if shard_path not in shard_groups:
                shard_groups[shard_path] = []
            for j in range(n):
                shard_groups[shard_path].append(j)

        shard_list = list(shard_groups.items())

        if shuffle_shards:
            random.shuffle(shard_list)

        for shard_path, local_indices in shard_list:
            for j in local_indices:
                self.samples.append((shard_path, j))

        self._cache       = {}
        self._cache_order = []
        self._max_cache   = max_cache_shards

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shard_path, local_idx = self.samples[idx]

        if shard_path not in self._cache:
            data = torch.load(shard_path, weights_only=True)

            if len(self._cache_order) >= self._max_cache:
                old_shard = self._cache_order.pop(0)
                del self._cache[old_shard]
                gc.collect()

            self._cache[shard_path]  = data
            self._cache_order.append(shard_path)

        return self._cache[shard_path][local_idx]

def aee_loss(pred, target, valid_mask=None):
    epe = torch.sqrt(
        (pred[:, 0] - target[:, 0])**2 +
        (pred[:, 1] - target[:, 1])**2 +
        1e-6
    )
    if valid_mask is not None:
        epe = epe[valid_mask.bool()]
    return epe.mean()

def compute_metrics(pred, target, valid_mask=None):
    with torch.no_grad():
        error = pred - target

        epe  = torch.sqrt(error[:, 0]**2 + error[:, 1]**2 + 1e-6)
        npe1 = (epe > 1.0).float()
        npe2 = (epe > 2.0).float()
        npe3 = (epe > 3.0).float()

        pred_u = pred[:, 0];   pred_v = pred[:, 1]
        gt_u   = target[:, 0]; gt_v   = target[:, 1]
        dot       = pred_u * gt_u + pred_v * gt_v + 1.0
        norm_pred = torch.sqrt(pred_u**2 + pred_v**2 + 1.0)
        norm_gt   = torch.sqrt(gt_u**2   + gt_v**2   + 1.0)
        cos_angle = torch.clamp(dot / (norm_pred * norm_gt), -1.0, 1.0)
        ae        = torch.acos(cos_angle) * (180.0 / torch.pi)

        if valid_mask is not None:
            mask = valid_mask.bool()
            epe  = epe[mask]
            ae   = ae[mask]
            npe1 = npe1[mask]
            npe2 = npe2[mask]
            npe3 = npe3[mask]

        return {
            "EPE": epe.mean().item(),
            "AE":  ae.mean().item(),
            "1PE": npe1.mean().item() * 100.0,
            "2PE": npe2.mean().item() * 100.0,
            "3PE": npe3.mean().item() * 100.0,
        }

class EventFlowUNet(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.enc1       = self.conv_block(in_channels, 64)
        self.enc2       = self.conv_block(64,          128)
        self.enc3       = self.conv_block(128,         256)
        self.enc4       = self.conv_block(256,         512)
        self.bottleneck = self.conv_block(512,         1024)
        self.dec4       = self.conv_block(1024 + 512,  512)
        self.dec3       = self.conv_block(512  + 256,  256)
        self.dec2       = self.conv_block(256  + 128,  128)
        self.dec1       = self.conv_block(128  + 64,    64)
        self.out        = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b  = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([F.interpolate(b,  e4.shape[2:], mode='bilinear', align_corners=False), e4], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, e3.shape[2:], mode='bilinear', align_corners=False), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False), e1], 1))
        return self.out(d1)

def checkpoint_path(run_name):
    return os.path.join(MODEL_SAVE_DIR, f"{run_name}_checkpoint.pt")

def save_checkpoint(run_name, epoch, model, optimizer, scaler, scheduler, best_val_loss, history):
    path = checkpoint_path(run_name)
    torch.save({
        "epoch":          epoch,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scaler":         scaler.state_dict(),
        "scheduler":      scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss":  best_val_loss,
        "history":        history,
    }, path)
    print(f"  💾 Checkpoint saved → {path}")

def load_checkpoint(run_name, model, optimizer, scaler, scheduler, device):
    path = checkpoint_path(run_name)
    if not os.path.isfile(path):
        return 0, float("inf"), []

    print(f"\n  🔄 Checkpoint found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch    = ckpt["epoch"] + 1
    best_val_loss  = ckpt["best_val_loss"]
    history        = ckpt.get("history", [])

    print(f"  ✓ Resuming from epoch {start_epoch} | Best val AEE so far: {best_val_loss:.4f} px")
    return start_epoch, best_val_loss, history

SEP  = "=" * 70
SEP2 = "-" * 70

def discover_runs(runs_dir, cache_dir):
    run_names = set()

    if os.path.isdir(runs_dir):
        for d in os.listdir(runs_dir):
            if os.path.isdir(os.path.join(runs_dir, d)):
                run_names.add(d)

    if os.path.isdir(cache_dir):
        for entry in os.listdir(cache_dir):
            full_path = os.path.join(cache_dir, entry)

            if os.path.isdir(full_path):
                run_names.add(entry)
            elif entry.endswith((".pt", ".npy")):
                base = entry.rsplit(".", 1)[0]
                if "_shard" in base:
                    name = base[: base.rfind("_shard")]
                    if name:
                        run_names.add(name)

    if not run_names:
        return []

    runs = []
    for name in sorted(run_names):
        run_dir       = os.path.join(runs_dir, name)
        has_raw_dir   = os.path.isdir(run_dir)
        shards, total = count_shards(cache_dir, name)
        runs.append({
            "name":        name,
            "dir":         run_dir if has_raw_dir else None,
            "has_raw_dir": has_raw_dir,
            "cached":      len(shards) > 0,
            "num_shards":  len(shards),
            "num_samples": total,
        })
    return runs


def discover_dsec_runs(dsec_cache_dir):
    if not os.path.isdir(dsec_cache_dir):
        return []

    sequences = {}
    for fname in sorted(os.listdir(dsec_cache_dir)):
        if not fname.endswith(".pt"):
            continue
        base = fname[: fname.rfind("_shard")]
        if not base:
            continue
        sequences.setdefault(base, []).append(os.path.join(dsec_cache_dir, fname))

    runs = []
    for name, shard_paths in sorted(sequences.items()):
        total = 0
        for p in shard_paths:
            try:
                data   = torch.load(p, weights_only=True)
                total += len(data)
            except Exception:
                pass
        runs.append({
            "name":        name,
            "dir":         dsec_cache_dir,
            "cached":      True,
            "num_shards":  len(shard_paths),
            "num_samples": total,
            "shard_paths": shard_paths,
        })
    return runs


def print_run_table(runs):
    print(SEP)
    print("                    CARLA RUNS - SELECTION MENU")
    print(SEP)
    print(f"{'IDX':<6} {'RUN NAME':<42} {'STATUS':<22} {'SAMPLES'}")
    print(SEP2)
    for i, r in enumerate(runs):
        if r["cached"] and not r.get("has_raw_dir", True):
            status = "✓ Cached (raw deleted)"
        elif r["cached"]:
            status = "✓ Cached"
        else:
            status = "✗ Not cached"
        print(f"{i:<6} {r['name']:<42} {status:<22} {r['num_samples'] if r['cached'] else '-'}")
    print(SEP2)
    print(f"Total: {len(runs)} runs available")
    print(SEP)


def parse_selection(selection_str, max_idx):
    s = selection_str.strip().lower()
    if s == "all":
        return list(range(max_idx + 1))

    indices = set()
    parts   = [p.strip() for p in s.split(",")]

    for part in parts:
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                return None
            try:
                lo, hi = int(bounds[0]), int(bounds[1])
            except ValueError:
                return None
            if lo > hi or lo < 0 or hi > max_idx:
                return None
            indices.update(range(lo, hi + 1))
        else:
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx > max_idx:
                return None
            indices.add(idx)

    return sorted(indices)


def interactive_run_selection(runs):
    print_run_table(runs)

    print("SELECTION OPTIONS:")
    print("  • Single index:      5")
    print("  • Multiple indices:  0,3,5,10")
    print("  • Range:             0-5")
    print("  • Combined:          0-3,7,10-15")
    print("  • All runs:          all")
    print("  • Quit:              q")
    print(SEP2)

    while True:
        raw = input(">>> Select runs: ").strip()

        if raw.lower() == "q":
            print("Quitting.")
            raise SystemExit(0)

        if raw.lower() == "all":
            selected_indices = list(range(len(runs)))
            print(f"✓ Selected ALL {len(runs)} runs")
        else:
            selected_indices = parse_selection(raw, len(runs) - 1)
            if selected_indices is None:
                print(f"  ✗ Invalid selection '{raw}'. Try again.")
                continue
            print(f"✓ Selected {len(selected_indices)} run(s): {selected_indices}")

        confirm = input("Confirm? (y/n): ").strip().lower()
        if confirm == "y":
            return [runs[i] for i in selected_indices]
        else:
            print("  Selection cancelled. Try again.\n")
            print_run_table(runs)
            print(SEP2)


def interactive_dsec_run_selection(dsec_runs):
    print(SEP)
    print("                   DSEC RUNS - SELECTION MENU")
    print(SEP)
    print(f"{'IDX':<6} {'SEQUENCE NAME':<42} {'SHARDS':<10} {'SAMPLES'}")
    print(SEP2)
    for i, r in enumerate(dsec_runs):
        print(f"{i:<6} {r['name']:<42} {r['num_shards']:<10} {r['num_samples']}")
    print(SEP2)
    print(f"Total: {len(dsec_runs)} DSEC sequences available")
    print(SEP)

    print("SELECTION OPTIONS:")
    print("  • Single index:      5")
    print("  • Multiple indices:  0,3,5,10")
    print("  • Range:             0-5")
    print("  • Combined:          0-3,7,10-15")
    print("  • All sequences:     all")
    print("  • Skip DSEC:         q")
    print(SEP2)

    while True:
        raw = input(">>> Select DSEC sequences (q to skip): ").strip()

        if raw.lower() == "q":
            print("  Skipping DSEC data.")
            return []

        if raw.lower() == "all":
            selected_indices = list(range(len(dsec_runs)))
            print(f"✓ Selected ALL {len(dsec_runs)} DSEC sequences")
        else:
            selected_indices = parse_selection(raw, len(dsec_runs) - 1)
            if selected_indices is None:
                print(f"  ✗ Invalid selection '{raw}'. Try again.")
                continue
            print(f"✓ Selected {len(selected_indices)} DSEC sequence(s): {selected_indices}")

        confirm = input("Confirm? (y/n): ").strip().lower()
        if confirm == "y":
            return [dsec_runs[i] for i in selected_indices]
        else:
            print("  Selection cancelled. Try again.\n")


def load_selected_dsec_runs(selected_dsec_runs):
    print(SEP)
    print("LOADING DSEC DATA")
    print(SEP)

    index = []
    for i, run in enumerate(selected_dsec_runs):
        print(f"[{i+1}/{len(selected_dsec_runs)}] {run['name']}")
        for p in run["shard_paths"]:
            try:
                data = torch.load(p, weights_only=True)
                index.append((p, len(data)))
                print(f"         ✓ {os.path.basename(p)}: {len(data)} samples")
            except Exception as e:
                print(f"         ✗ Failed to load {p}: {e}")

    total_samples = sum(n for _, n in index)
    print(SEP2)
    print(f"DSEC TOTAL: {total_samples} samples from {len(selected_dsec_runs)} sequence(s)")
    print(SEP)
    return index


def load_selected_runs(selected_runs, cache_dir):
    print(SEP)
    print("LOADING DATA")
    print(SEP)

    index = []
    total_runs = len(selected_runs)

    for i, run in enumerate(selected_runs):
        run_name = run["name"]
        print(f"[{i+1}/{total_runs}] {run_name}")

        shards, total = count_shards(cache_dir, run_name)

        if shards:
            print(f"         ✓ Loaded from cache: {total} samples ({len(shards)} shards)")
            for p in shards:
                data = torch.load(p, weights_only=True)
                index.append((p, len(data)))
        else:
            print(f"         ✗ No cache found. Run prepare_carla.py first to cache this run.")

    total_samples = sum(n for _, n in index)
    print(SEP2)
    print(f"TOTAL: {total_samples} samples from {total_runs} runs")
    print(SEP)

    return index


def ask_int(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  Invalid input, using default: {default}")
        return default


def ask_float(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid input, using default: {default}")
        return default


def ask_str(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw != "" else default


def ask_seed(prompt, default):
    env_val = os.environ.get("TRAIN_SEED")
    if env_val is not None:
        try:
            seed = int(env_val)
            print(f"{prompt}: {seed}  (from TRAIN_SEED env var)")
            return seed
        except ValueError:
            print(f"  Invalid TRAIN_SEED='{env_val}', falling back to prompt.")
    return ask_int(prompt, default)


def interactive_training_config():
    print(SEP)
    print("TRAINING CONFIGURATION")
    print(SEP)
    print("Press Enter to use default values shown in [brackets]")
    print(SEP2)

    epochs             = ask_int  ("Epochs",                           50)
    batch_size         = ask_int  ("Batch size",                        8)
    lr                 = ask_float("Learning rate",                   1e-4)
    val_split          = ask_float("Validation split (fraction)",      0.2)
    accumulation_steps = ask_int  ("Gradient accumulation steps",       2)
    num_workers        = ask_int  ("DataLoader num_workers",            3)
    max_cache_shards   = ask_int  ("Max shards in RAM (train dataset)", 15)
    run_name           = ask_str  ("Run name (used for saved model)", "unet_flow")
    seed                = ask_seed("Random seed (weight init / batch order)", 0)

    if f"_seed{seed}" not in run_name:
        run_name = f"{run_name}_seed{seed}"

    print(SEP2)
    print("LR SCHEDULER  (ReduceLROnPlateau — reduces LR when val loss stalls)")
    scheduler_patience = ask_int  ("  Patience (epochs before reducing LR)", 5)
    scheduler_factor   = ask_float("  Factor (LR multiplier on reduction)",  0.5)
    min_lr             = ask_float("  Min LR",                              1e-7)

    print(SEP2)
    ckpt_every         = ask_int  ("Save checkpoint every N epochs (0 = only on best)", 1)

    print(SEP2)
    print("Configuration summary:")
    print(f"  Epochs:             {epochs}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Learning rate:      {lr}")
    print(f"  Val split:          {val_split:.0%}")
    print(f"  Accum steps:        {accumulation_steps}")
    print(f"  Num workers:        {num_workers}")
    print(f"  Max cache shards:   {max_cache_shards}")
    print(f"  Run name:           {run_name}")
    print(f"  Seed:               {seed}  (split seed fixed at {SPLIT_SEED} across all seeds)")
    print(f"  LR scheduler:       patience={scheduler_patience}, factor={scheduler_factor}, min_lr={min_lr}")
    print(f"  Checkpoint every:   {ckpt_every} epoch(s)")
    print(SEP2)

    confirm = input("Start training? (y/n): ").strip().lower()
    if confirm != "y":
        print("Training cancelled.")
        raise SystemExit(0)

    return {
        "epochs":              epochs,
        "batch_size":          batch_size,
        "lr":                  lr,
        "val_split":           val_split,
        "accumulation_steps":  accumulation_steps,
        "num_workers":         num_workers,
        "max_cache_shards":    max_cache_shards,
        "run_name":            run_name,
        "seed":                seed,
        "split_seed":          SPLIT_SEED,
        "scheduler_patience":  scheduler_patience,
        "scheduler_factor":    scheduler_factor,
        "min_lr":              min_lr,
        "ckpt_every":          ckpt_every,
    }


def train(index, cfg, device):
    index = list(index)
    random.Random(SPLIT_SEED).shuffle(index)
    split     = int((1.0 - cfg["val_split"]) * len(index))
    train_idx = index[:split]
    val_idx   = index[split:]

    set_seed(cfg["seed"])

    train_ds = EventFlowDataset(train_idx, shuffle_shards=True,  max_cache_shards=cfg["max_cache_shards"])
    val_ds   = EventFlowDataset(val_idx,   shuffle_shards=False, max_cache_shards=max(cfg["max_cache_shards"] // 2, 5))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=max(cfg["num_workers"] - 1, 1),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"\nTrain: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    model     = EventFlowUNet(in_channels=5).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=1e-4,
    )
    scaler    = GradScaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg["scheduler_patience"],
        factor=cfg["scheduler_factor"],
        min_lr=cfg["min_lr"],
    )

    start_epoch, best_val_loss, history = load_checkpoint(
        cfg["run_name"], model, optimizer, scaler, scheduler, device
    )

    accumulation_steps = cfg["accumulation_steps"]
    best_model_path    = os.path.join(MODEL_SAVE_DIR, f"{cfg['run_name']}_best.pt")
    ckpt_every         = cfg.get("ckpt_every", 1)

    print(f"\nTraining started! Model will be saved to: {MODEL_SAVE_DIR}\n")

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_start = time.time()
        print(SEP)
        print(f"EPOCH {epoch + 1} / {cfg['epochs']}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(SEP)

        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (voxels, flows, valids) in enumerate(train_loader):
            voxels = voxels.to(device, non_blocking=True)
            flows  = flows.to(device,  non_blocking=True)
            valids = valids.to(device, non_blocking=True)

            with autocast("cuda"):
                pred = model(voxels)
                loss = aee_loss(pred, flows, valids) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * accumulation_steps

            if batch_idx % 20 == 0:
                if device.type == "cuda":
                    mem = torch.cuda.memory_reserved(device) / (1024 ** 3)
                    print(f"  Batch {batch_idx:>4} | AEE: {loss.item()*accumulation_steps:.4f} px | VRAM: {mem:.1f} GB")
                else:
                    print(f"  Batch {batch_idx:>4} | AEE: {loss.item()*accumulation_steps:.4f} px")

        remainder = len(train_loader) % accumulation_steps
        if remainder != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_loss    = 0.0
        metrics_sum = {"EPE": 0.0, "AE": 0.0, "1PE": 0.0, "2PE": 0.0, "3PE": 0.0}

        with torch.no_grad():
            for voxels, flows, valids in val_loader:
                voxels = voxels.to(device, non_blocking=True)
                flows  = flows.to(device,  non_blocking=True)
                valids = valids.to(device, non_blocking=True)
                with autocast("cuda"):
                    pred      = model(voxels)
                    val_loss += aee_loss(pred, flows, valids).item()
                    m         = compute_metrics(pred, flows, valids)
                for k in metrics_sum:
                    metrics_sum[k] += m[k]

        avg_train   = train_loss / len(train_loader)
        avg_val     = val_loss   / len(val_loader)
        avg_metrics = {k: v / len(val_loader) for k, v in metrics_sum.items()}
        epoch_time  = time.time() - epoch_start

        print(f"\n  Train AEE: {avg_train:.4f} px | Val AEE: {avg_val:.4f} px | Time: {epoch_time:.0f}s")
        print(f"  EPE: {avg_metrics['EPE']:.4f} px | "
              f"AE: {avg_metrics['AE']:.2f} deg | "
              f"1PE: {avg_metrics['1PE']:.1f}% | "
              f"2PE: {avg_metrics['2PE']:.1f}% | "
              f"3PE: {avg_metrics['3PE']:.1f}%")

        history.append({
            "epoch":      epoch + 1,
            "train_loss": avg_train,
            "val_loss":   avg_val,
            "lr":         optimizer.param_groups[0]["lr"],
            **avg_metrics,
        })

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved (Val AEE: {avg_val:.4f} px)")

        if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
            save_checkpoint(
                cfg["run_name"], epoch,
                model, optimizer, scaler, scheduler,
                best_val_loss, history
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_path = os.path.join(MODEL_SAVE_DIR, f"{cfg['run_name']}_final.pt")
    torch.save(model.state_dict(), final_path)

    history_path = os.path.join(MODEL_SAVE_DIR, f"{cfg['run_name']}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    config_path = os.path.join(MODEL_SAVE_DIR, f"{cfg['run_name']}_config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{SEP}")
    print(f"Training complete!")
    print(f"  Best model  : {best_model_path}")
    print(f"  Final model : {final_path}")
    print(f"  Checkpoint  : {checkpoint_path(cfg['run_name'])}")
    print(f"  History     : {history_path}")
    print(SEP)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(SEP)
    print("        CARLA + DSEC OPTICAL FLOW TRAINING - INTERACTIVE MODE")
    print(SEP)
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  VRAM        : {vram:.1f} GB")
    print(f"  Runs dir    : {CARLA_RUNS_DIR}")
    print(f"  Cache dir   : {CACHE_DIR}")
    print(f"  DSEC cache  : {DSEC_CACHE_DIR}")
    print(f"  Models dir  : {MODEL_SAVE_DIR}")
    print(SEP)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,      exist_ok=True)

    print(SEP)
    print("DATA SOURCE")
    print(SEP)
    print("  1  CARLA only")
    print("  2  DSEC only")
    print("  3  CARLA + DSEC (mixed)")
    print(SEP2)
    while True:
        src = input(">>> Data source [1]: ").strip()
        if src == "":
            src = "1"
        if src in ("1", "2", "3"):
            break
        print("  ✗ Enter 1, 2, or 3.")

    use_carla = src in ("1", "3")
    use_dsec  = src in ("2", "3")
    index     = []

    if use_carla:
        runs = discover_runs(CARLA_RUNS_DIR, CACHE_DIR)
        if not runs:
            print(f"No CARLA runs found in: {CARLA_RUNS_DIR}")
            raise SystemExit(1)
        selected_runs  = interactive_run_selection(runs)
        carla_index    = load_selected_runs(selected_runs, CACHE_DIR)
        if not carla_index:
            print("No cached data available for the selected CARLA runs.")
            print("Run prepare_carla.py first to process and cache your data.")
            raise SystemExit(1)
        index.extend(carla_index)

    if use_dsec:
        dsec_runs = discover_dsec_runs(DSEC_CACHE_DIR)
        if not dsec_runs:
            print(f"  ✗ No DSEC sequences found in: {DSEC_CACHE_DIR}")
            if not index:
                raise SystemExit(1)
            print("    Continuing with CARLA data only.")
        else:
            selected_dsec = interactive_dsec_run_selection(dsec_runs)
            if selected_dsec:
                dsec_index = load_selected_dsec_runs(selected_dsec)
                index.extend(dsec_index)

    if not index:
        print("No data selected. Exiting.")
        raise SystemExit(1)

    print(f"\n  ✓ Total dataset: {sum(n for _, n in index)} samples")

    cfg = interactive_training_config()

    train(index, cfg, device)

