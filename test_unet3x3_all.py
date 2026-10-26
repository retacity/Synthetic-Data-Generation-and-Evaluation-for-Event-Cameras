import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
import os
import gc
import cv2
import csv
import time
from pathlib import Path
from tqdm import tqdm

CACHE_DIR  = r"C:\Users\Reta-Fixo\Tese\models\dsec_cache"
MODELS_DIR = r"C:\Users\Reta-Fixo\Tese\models\TrainModels"
OUTPUT_DIR = r"C:\Users\Reta-Fixo\Tese\models\dsec_results"

EXTRA_MODEL_DIRS = [
    r"C:\Users\Reta-Fixo\Tese\models",
]

CONDITION_OVERRIDES = {
    "try1_carla+dsec_best.pt": "mix",
}

H_OUT, W_OUT = 480, 640
B_BINS       = 5
BATCH_SIZE   = 8
NUM_WORKERS  = 2

SEQUENCES_TO_TEST = [
    "thun_00_a",
    "zurich_city_01_a",
    "zurich_city_08_a",
    "zurich_city_11_a",
    "zurich_city_11_c",
]

MODELS_TO_TEST = [
    "seed1_carla_seed1_best.pt",
    "seed1_dsec_seed1_best.pt",
    "seed1_mix_seed1_best.pt",
    "seed2_carla_seed2_best.pt",
    "seed2_dsec_seed2_best.pt",
    "seed2_mix_seed2_best.pt",
    "seed3_carla_seed3_best.pt",
    "seed3_dsec_seed3_best.pt",
    "try1_carla+dsec_best.pt",
]

MAG_BINS = [
    ("slow",   0.0,  2.0),
    ("medium", 2.0, 10.0),
    ("fast",  10.0, float("inf")),
]
MAG_KEYS = [f"EPE_{name}" for name, _, _ in MAG_BINS]

SAVE_PREDICTIONS = True

SEP  = "=" * 70
SEP2 = "-" * 70


def parse_selection(selection_str: str, max_idx: int) -> list[int] | None:
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


def discover_models(models_dir: str, extra_dirs: list[str] | None = None) -> list[dict]:
    seen: dict[str, dict] = {}

    def add(p: Path):
        resolved = str(p.resolve())
        if resolved in seen:
            return
        size_mb = p.stat().st_size / (1024 ** 2)
        try:
            rel = str(p.relative_to(models_dir))
        except ValueError:
            rel = str(p)
        seen[resolved] = {"name": rel, "path": str(p), "size_mb": size_mb}

    for p in Path(models_dir).rglob("*.pt"):
        add(p)

    for extra in (extra_dirs or []):
        extra_path = Path(extra)
        if not extra_path.is_dir():
            continue
        for p in extra_path.glob("*.pt"):
            add(p)

    return [seen[k] for k in sorted(seen, key=lambda k: seen[k]["name"])]


def print_model_table(models: list[dict]):
    print(SEP)
    print("                    AVAILABLE MODELS - SELECTION MENU")
    print(SEP)
    print(f"{'IDX':<6} {'MODEL PATH':<50} {'SIZE':>8}")
    print(SEP2)
    for i, m in enumerate(models):
        print(f"{i:<6} {m['name']:<50} {m['size_mb']:>6.1f} MB")
    print(SEP2)
    print(f"Total: {len(models)} model(s) found in {MODELS_DIR}")
    print(SEP)


def interactive_model_selection(models: list[dict]) -> list[dict]:
    print_model_table(models)
    print("SELECTION OPTIONS:")
    print("  • Single index:      3")
    print("  • Multiple indices:  0,3,5,10")
    print("  • Range:             0-5")
    print("  • Combined:          0-3,7,10-15")
    print("  • All models:        all")
    print("  • Quit:              q")
    print(SEP2)

    while True:
        raw = input(">>> Select model(s): ").strip()

        if raw.lower() == "q":
            print("Quitting.")
            raise SystemExit(0)

        if raw.lower() == "all":
            selected_indices = list(range(len(models)))
        else:
            selected_indices = parse_selection(raw, len(models) - 1)
            if selected_indices is None:
                print(f"  ✗ Invalid selection '{raw}'. Try again.")
                continue

        chosen = [models[i] for i in selected_indices]
        print(f"  ✓ Selected {len(chosen)} model(s):")
        for m in chosen:
            print(f"      - {m['name']}  ({m['size_mb']:.1f} MB)")

        confirm = input("Confirm? (y/n): ").strip().lower()
        if confirm == "y":
            return chosen
        else:
            print("  Selection cancelled. Try again.\n")
            print_model_table(models)
            print(SEP2)


def select_models_by_name(models: list[dict], names: list[str]) -> list[dict]:
    by_relpath  = {m["name"]: m for m in models}
    by_filename = {Path(m["name"]).name: m for m in models}

    chosen, missing = [], []
    for n in names:
        if n in by_relpath:
            chosen.append(by_relpath[n])
        elif n in by_filename:
            chosen.append(by_filename[n])
        else:
            missing.append(n)

    if missing:
        print(f"  ⚠ Not found in {MODELS_DIR} (skipped): {', '.join(missing)}")
    if not chosen:
        print(f"  ✗ None of MODELS_TO_TEST were found under {MODELS_DIR}.")
        raise SystemExit(1)

    print(f"  ✓ Hard-coded selection — {len(chosen)} model(s):")
    for m in chosen:
        print(f"      - {m['name']}  ({m['size_mb']:.1f} MB)")
    return chosen


def discover_sequences(cache_dir: str) -> list[dict]:
    raw: dict[str, list[str]] = {}
    for p in sorted(Path(cache_dir).glob("*_shard*.pt")):
        stem     = p.stem
        idx_part = stem.rsplit("_shard", 1)
        if len(idx_part) != 2:
            continue
        seq_name = idx_part[0]
        raw.setdefault(seq_name, []).append(str(p))

    seqs = []
    for name, shard_paths in sorted(raw.items()):
        total = 0
        for sp in shard_paths:
            try:
                data   = torch.load(sp, weights_only=True)
                total += len(data)
            except Exception:
                pass
        seqs.append({
            "name":        name,
            "shard_paths": shard_paths,
            "num_shards":  len(shard_paths),
            "num_samples": total,
        })
    return seqs


class EventFlowUNet(nn.Module):
    def __init__(self, in_channels: int = 5):
        super().__init__()
        self.enc1       = self._block(in_channels,  64)
        self.enc2       = self._block(64,           128)
        self.enc3       = self._block(128,          256)
        self.enc4       = self._block(256,          512)
        self.bottleneck = self._block(512,          1024)
        self.dec4       = self._block(1024 + 512,   512)
        self.dec3       = self._block(512  + 256,   256)
        self.dec2       = self._block(256  + 128,   128)
        self.dec1       = self._block(128  + 64,     64)
        self.out        = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def _block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class DSECCacheDataset(Dataset):

    def __init__(self, shard_paths: list[str], max_cache_shards: int = 10):
        self.samples: list[tuple[str, int]] = []

        for p in shard_paths:
            data = torch.load(p, weights_only=True)
            for j in range(len(data)):
                self.samples.append((p, j))

        self._cache:       dict[str, list] = {}
        self._cache_order: list[str]       = []
        self._max_cache    = max_cache_shards

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        shard_path, local_idx = self.samples[idx]

        if shard_path not in self._cache:
            if len(self._cache_order) >= self._max_cache:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
                gc.collect()
            self._cache[shard_path]  = torch.load(shard_path, weights_only=True)
            self._cache_order.append(shard_path)

        return self._cache[shard_path][local_idx]


def compute_metrics(pred: torch.Tensor,
                    target: torch.Tensor,
                    valid_mask: torch.Tensor | None = None) -> dict:
    with torch.no_grad():
        error = pred - target
        epe   = torch.sqrt(error[:, 0]**2 + error[:, 1]**2 + 1e-6)
        npe1  = (epe > 1.0).float()
        npe2  = (epe > 2.0).float()
        npe3  = (epe > 3.0).float()

        pu, pv  = pred[:, 0],   pred[:, 1]
        gu, gv  = target[:, 0], target[:, 1]
        dot       = pu * gu + pv * gv + 1.0
        norm_pred = torch.sqrt(pu**2 + pv**2 + 1.0)
        norm_gt   = torch.sqrt(gu**2 + gv**2 + 1.0)
        cos_a     = torch.clamp(dot / (norm_pred * norm_gt), -1.0, 1.0)
        ae        = torch.acos(cos_a) * (180.0 / torch.pi)

        gt_mag = torch.sqrt(gu**2 + gv**2)

        if valid_mask is not None:
            m      = valid_mask.bool()
            epe    = epe[m]
            ae     = ae[m]
            npe1   = npe1[m]
            npe2   = npe2[m]
            npe3   = npe3[m]
            gt_mag = gt_mag[m]

        out = {
            "EPE":      epe.mean().item(),
            "AE":       ae.mean().item(),
            "1PE":      npe1.mean().item() * 100.0,
            "2PE":      npe2.mean().item() * 100.0,
            "3PE":      npe3.mean().item() * 100.0,
            "n_pixels": epe.numel(),
        }
        for name, lo, hi in MAG_BINS:
            sel = (gt_mag >= lo) & (gt_mag < hi)
            out[f"EPEsum_{name}"] = epe[sel].sum().item()
            out[f"n_{name}"]      = int(sel.sum().item())
        return out


_METRIC_KEYS = ["EPE", "AE", "1PE", "2PE", "3PE"]


def infer_condition(model_name: str) -> str:
    fname = Path(model_name).name
    if fname in CONDITION_OVERRIDES:
        return CONDITION_OVERRIDES[fname]
    n = model_name.lower()
    if "mix" in n:
        return "mix"
    if "dsec" in n:
        return "dsec"
    if "carla" in n:
        return "carla"
    return "unknown"


def mean_std_by_condition(rows: list[dict], group_keys: list[str],
                          metric_keys: list[str] | None = None) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        cond   = infer_condition(r["model"])
        gkey   = (cond,) + tuple(r[k] for k in group_keys)
        groups.setdefault(gkey, []).append(r)

    out = []
    for gkey, runs in groups.items():
        cond = gkey[0]
        n    = len(runs)
        row: dict = {"condition": cond}
        row.update(dict(zip(group_keys, gkey[1:])))
        row["n_seeds"] = n
        row["seed_models"] = [r["model"] for r in runs]
        for key in (metric_keys or _METRIC_KEYS):
            vals = np.array([r[key] for r in runs], dtype=float)
            row[f"{key}_mean"] = float(vals.mean())
            row[f"{key}_std"]  = float(vals.std(ddof=1)) if n > 1 else float("nan")
        out.append(row)

    cond_order = {"carla": 0, "dsec": 1, "mix": 2, "unknown": 3}
    out.sort(key=lambda r: (cond_order.get(r["condition"], 99),) +
                            tuple(r[k] for k in group_keys))
    return out


def save_flow_png(flow_np: np.ndarray, out_path: str):
    H, W = flow_np.shape[1], flow_np.shape[2]
    flow_np = flow_np.astype(np.float32, copy=False)
    img  = np.zeros((H, W, 3), dtype=np.uint16)
    img[:, :, 2] = np.clip(flow_np[0] * 128.0 + 2**15, 0, 65535).astype(np.uint16)
    img[:, :, 1] = np.clip(flow_np[1] * 128.0 + 2**15, 0, 65535).astype(np.uint16)
    img[:, :, 0] = 1
    cv2.imwrite(str(out_path), img)


def evaluate_sequence(seq: dict, model: nn.Module,
                      device: torch.device, output_dir: str,
                      model_name: str, save_predictions: bool = True,
                      verbose: bool = False) -> dict | None:
    seq_name    = seq["name"]
    shard_paths = seq["shard_paths"]

    if save_predictions:
        seq_out = os.path.join(output_dir, model_name, seq_name)
        os.makedirs(seq_out, exist_ok=True)

    dataset = DSECCacheDataset(shard_paths, max_cache_shards=8)
    loader  = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = device.type == 'cuda',
    )

    all_metrics: list[dict] = []
    use_amp   = device.type == 'cuda'
    frame_idx = 0
    seq_start = time.time()

    for batch in tqdm(loader, desc=f"  {seq_name}", ncols=70, leave=False):
        voxels = batch[0]
        flows  = batch[1]
        if len(batch) > 2 and batch[2] is not None:
            valids = batch[2]
        else:
            raise RuntimeError(
                "Shard is missing the valid mask (3rd element). "
                "Delete your dsec_cache folder and re-run prepare_dsec.py."
            )

        voxels = voxels.to(device, non_blocking=True)
        flows  = flows.to(device,  non_blocking=True)
        valids = valids.to(device, non_blocking=True)

        with torch.no_grad():
            if use_amp:
                with autocast('cuda'):
                    preds = model(voxels)
                preds = preds.float()
            else:
                preds = model(voxels)

        for b in range(preds.shape[0]):
            pred_2d  = preds[b]
            flow_2d  = flows[b]
            valid_2d = valids[b]

            pred_flat  = pred_2d.permute(1, 2, 0).reshape(-1, 2)
            flow_flat  = flow_2d.permute(1, 2, 0).reshape(-1, 2)
            valid_flat = valid_2d.reshape(-1)

            m = compute_metrics(pred_flat, flow_flat, valid_flat)
            all_metrics.append(m)

            if save_predictions:
                pred_np = pred_2d.cpu().numpy()
                save_flow_png(pred_np, os.path.join(seq_out, f"{frame_idx:06d}.png"))
            frame_idx += 1

        del voxels, flows, preds

    if not all_metrics:
        return None

    total_px  = sum(m["n_pixels"] for m in all_metrics)
    agg = {k: sum(m[k] * m["n_pixels"] for m in all_metrics) / total_px
           for k in ["EPE", "AE", "1PE", "2PE", "3PE"]}
    for name, _, _ in MAG_BINS:
        agg[f"EPEsum_{name}"] = sum(m[f"EPEsum_{name}"] for m in all_metrics)
        agg[f"n_{name}"]      = sum(m[f"n_{name}"]      for m in all_metrics)
        n = agg[f"n_{name}"]
        agg[f"EPE_{name}"]    = agg[f"EPEsum_{name}"] / n if n > 0 else float("nan")
    elapsed   = time.time() - seq_start

    if verbose:
        print(f"\n  [{seq_name}]  {len(all_metrics)} frames | {total_px:,} valid pixels | {elapsed:.0f}s")
        print(f"    EPE: {agg['EPE']:.4f} px  |  AE: {agg['AE']:.2f} deg  |  "
              f"1PE: {agg['1PE']:.2f}%  |  2PE: {agg['2PE']:.2f}%  |  3PE: {agg['3PE']:.2f}%")

    return {"seq": seq_name, "n_frames": len(all_metrics),
            "n_pixels": total_px, **agg}


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(SEP)
    print("          DSEC OPTICAL FLOW EVALUATION - INTERACTIVE MODE")
    print(SEP)
    print(f"  Device      : {device}")
    if device.type == 'cuda':
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  VRAM        : {vram:.1f} GB")
    print(f"  Cache dir   : {CACHE_DIR}")
    print(f"  Models dir  : {MODELS_DIR}")
    if EXTRA_MODEL_DIRS:
        print(f"  Extra dirs  : {', '.join(EXTRA_MODEL_DIRS)}  (top-level .pt files only)")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(SEP)

    models = discover_models(MODELS_DIR, EXTRA_MODEL_DIRS)
    if not models:
        print(f"  ✗ No .pt files found in {MODELS_DIR}")
        raise SystemExit(1)

    if MODELS_TO_TEST:
        print_model_table(models)
        chosen_models = select_models_by_name(models, MODELS_TO_TEST)
    else:
        chosen_models = interactive_model_selection(models)

    all_seqs     = discover_sequences(CACHE_DIR)
    seqs_by_name = {s["name"]: s for s in all_seqs}

    selected_seqs = [seqs_by_name[n] for n in SEQUENCES_TO_TEST if n in seqs_by_name]
    missing       = [n for n in SEQUENCES_TO_TEST if n not in seqs_by_name]

    if not selected_seqs:
        print(f"  ✗ None of the target sequences were found in {CACHE_DIR}")
        raise SystemExit(1)
    if missing:
        print(f"  ⚠ Not found in cache (skipped): {', '.join(missing)}")

    total_samples = sum(s["num_samples"] for s in selected_seqs)
    print(SEP)
    print(f"  Test sequences ({len(selected_seqs)}): {', '.join(s['name'] for s in selected_seqs)}")
    print(f"  Total dataset  : {total_samples} samples")
    print(f"  Save flow PNGs : {SAVE_PREDICTIONS}")
    print(SEP)

    model_summaries: list[dict] = []
    per_sequence_summaries: list[dict] = []

    for mi, chosen_model in enumerate(chosen_models):
        print(f"\n{SEP}")
        print(f"MODEL [{mi+1}/{len(chosen_models)}]  {chosen_model['name']}")
        print(SEP)

        model = EventFlowUNet(in_channels=B_BINS).to(device)
        ckpt  = torch.load(chosen_model["path"], map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()

        model_stem      = Path(chosen_model["name"]).stem
        model_safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_stem)

        eval_start   = time.time()
        seq_results: list[dict] = []

        for si, seq in enumerate(selected_seqs):
            print(f"  [{si+1}/{len(selected_seqs)}] {seq['name']}  "
                  f"({seq['num_shards']} shards, {seq['num_samples']} samples)")
            result = evaluate_sequence(seq, model, device, OUTPUT_DIR, model_safe_name,
                                        save_predictions=SAVE_PREDICTIONS, verbose=False)
            if result is not None:
                seq_results.append(result)
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if not seq_results:
            print(f"\n  ✗ No results produced for {chosen_model['name']}. Skipping.")
            continue

        print(f"\n{SEP2}")
        print(f"PER-SEQUENCE (TOWN) RESULTS  —  {chosen_model['name']}")
        print(SEP2)
        print(f"{'SEQUENCE':<22}{'FRAMES':>8}{'EPE':>9}{'AE':>8}{'1PE':>8}{'2PE':>8}{'3PE':>8}")
        for r in seq_results:
            print(f"{r['seq']:<22}{r['n_frames']:>8}{r['EPE']:>9.4f}{r['AE']:>8.2f}"
                  f"{r['1PE']:>7.2f}%{r['2PE']:>7.2f}%{r['3PE']:>7.2f}%")
        print(SEP2)

        for r in seq_results:
            per_sequence_summaries.append({
                "model":    chosen_model["name"],
                "sequence": r["seq"],
                "n_frames": r["n_frames"],
                "n_pixels": r["n_pixels"],
                "EPE": r["EPE"], "AE": r["AE"],
                "1PE": r["1PE"], "2PE": r["2PE"], "3PE": r["3PE"],
                **{k: r[k] for k in MAG_KEYS},
            })

        total_time = time.time() - eval_start
        total_px   = sum(r["n_pixels"] for r in seq_results)
        agg = {k: sum(r[k] * r["n_pixels"] for r in seq_results) / total_px
               for k in ["EPE", "AE", "1PE", "2PE", "3PE"]}
        for name, _, _ in MAG_BINS:
            n = sum(r[f"n_{name}"] for r in seq_results)
            agg[f"n_{name}"]   = n
            agg[f"EPE_{name}"] = (sum(r[f"EPEsum_{name}"] for r in seq_results) / n
                                  if n > 0 else float("nan"))

        print(f"\n{SEP2}")
        print(f"GLOBAL RESULTS  —  {chosen_model['name']}")
        print(f"  {len(seq_results)} sequences  |  {total_px:,} valid pixels  |  {total_time:.0f}s")
        for key, unit in [("EPE", "px"), ("AE", "deg"),
                          ("1PE", "%"), ("2PE", "%"), ("3PE", "%")]:
            print(f"  {key:>4s}: {agg[key]:.4f} {unit}")
        for name, lo, hi in MAG_BINS:
            rng = f"{lo:g}-{hi:g} px" if hi != float("inf") else f">={lo:g} px"
            print(f"  EPE {name:<6} ({rng:<9}): {agg[f'EPE_{name}']:.4f} px   "
                  f"[{agg[f'n_{name}']:,} px, {100.0 * agg[f'n_{name}'] / total_px:.1f}% of valid]")
        print(SEP2)

        model_summaries.append({
            "model":    chosen_model["name"],
            "n_seqs":   len(seq_results),
            "n_pixels": total_px,
            "time_s":   round(total_time, 1),
            **agg,
        })

    if not model_summaries:
        print("\n  ✗ No results produced for any model. Check your CACHE_DIR / MODELS_DIR.")
        raise SystemExit(1)

    print(f"\n{SEP}")
    print(f"GLOBAL RESULTS — ALL MODELS  ({len(model_summaries)} model(s), "
          f"sequences: {', '.join(s['name'] for s in selected_seqs)})")
    print(SEP)
    print(f"{'MODEL':<40} {'EPE':>8} {'AE':>8} {'1PE':>8} {'2PE':>8} {'3PE':>8}")
    print(SEP2)
    for r in model_summaries:
        print(f"{r['model']:<40} {r['EPE']:>8.4f} {r['AE']:>8.2f} "
              f"{r['1PE']:>7.2f}% {r['2PE']:>7.2f}% {r['3PE']:>7.2f}%")
    print(SEP)

    csv_path = os.path.join(OUTPUT_DIR, "global_results_all_models.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "n_seqs", "n_pixels", "time_s",
                           "EPE", "AE", "1PE", "2PE", "3PE"] + MAG_KEYS +
                          [f"n_{name}" for name, _, _ in MAG_BINS])
        writer.writeheader()
        writer.writerows(model_summaries)

    print(f"  CSV saved  : {csv_path}")

    def fmt_ms(mean: float, std: float, decimals: int = 2) -> str:
        if std != std:
            return f"{mean:.{decimals}f} (n=1)"
        return f"{mean:.{decimals}f} \u00b1 {std:.{decimals}f}"

    condition_summary = mean_std_by_condition(model_summaries, group_keys=[])

    print(f"\n{SEP}")
    print(f"CONDITION SUMMARY — MEAN \u00b1 STD ACROSS SEEDS")
    print(SEP)
    print(f"{'CONDITION':<12}{'N':>3}   {'EPE':<16}{'AE':<16}{'1PE':<16}{'2PE':<16}{'3PE':<16}")
    print(SEP2)
    for r in condition_summary:
        print(f"{r['condition']:<12}{r['n_seeds']:>3}   "
              f"{fmt_ms(r['EPE_mean'], r['EPE_std'], 4):<16}"
              f"{fmt_ms(r['AE_mean'],  r['AE_std'],  2):<16}"
              f"{fmt_ms(r['1PE_mean'], r['1PE_std'], 2):<16}"
              f"{fmt_ms(r['2PE_mean'], r['2PE_std'], 2):<16}"
              f"{fmt_ms(r['3PE_mean'], r['3PE_std'], 2):<16}")
    print(SEP2)
    for r in condition_summary:
        print(f"  {r['condition']} seeds: {', '.join(r['seed_models'])}")
    print(SEP)

    condition_csv_path = os.path.join(OUTPUT_DIR, "condition_mean_std.csv")
    with open(condition_csv_path, "w", newline="") as f:
        fieldnames = (["condition", "n_seeds"] +
                      [f"{k}_{stat}" for k in _METRIC_KEYS for stat in ("mean", "std")] +
                      ["seed_models"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in condition_summary:
            row = {k: r[k] for k in fieldnames if k != "seed_models"}
            row["seed_models"] = "; ".join(r["seed_models"])
            writer.writerow(row)

    print(f"  CSV saved  : {condition_csv_path}")

    print(f"\n{SEP}")
    print(f"PER-SEQUENCE (TOWN) RESULTS — ALL MODELS")
    print(SEP)
    print(f"{'MODEL':<32}{'SEQUENCE':<20}{'FRAMES':>8}{'EPE':>9}{'AE':>8}{'1PE':>8}{'2PE':>8}{'3PE':>8}")
    print(SEP2)
    for r in per_sequence_summaries:
        print(f"{r['model']:<32}{r['sequence']:<20}{r['n_frames']:>8}{r['EPE']:>9.4f}"
              f"{r['AE']:>8.2f}{r['1PE']:>7.2f}%{r['2PE']:>7.2f}%{r['3PE']:>7.2f}%")
    print(SEP)

    town_csv_path = os.path.join(OUTPUT_DIR, "per_sequence_results_all_models.csv")
    with open(town_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "sequence", "n_frames", "n_pixels",
                           "EPE", "AE", "1PE", "2PE", "3PE"] + MAG_KEYS)
        writer.writeheader()
        writer.writerows(per_sequence_summaries)

    print(f"  CSV saved  : {town_csv_path}")

    per_seq_condition_summary = mean_std_by_condition(per_sequence_summaries, group_keys=["sequence"])

    overall_rows = []
    for r in condition_summary:
        orow = dict(r)
        orow["sequence"] = "OVERALL"
        overall_rows.append(orow)

    _cond_order = {"carla": 0, "dsec": 1, "mix": 2, "unknown": 3}
    combined_summary = overall_rows + per_seq_condition_summary
    combined_summary.sort(key=lambda r: (
        _cond_order.get(r["condition"], 99),
        0 if r["sequence"] == "OVERALL" else 1,
        r["sequence"],
    ))

    print(f"\n{SEP}")
    print(f"PER-SEQUENCE CONDITION SUMMARY — MEAN \u00b1 STD ACROSS SEEDS")
    print(f"(OVERALL = pixel-weighted aggregate across all test sequences, matching Table VI)")
    print(SEP)
    print(f"{'CONDITION':<10}{'SEQUENCE':<20}{'N':>3}   {'EPE':<16}{'AE':<16}{'1PE':<16}{'2PE':<16}{'3PE':<16}")
    print(SEP2)
    for r in combined_summary:
        marker = ">>" if r["sequence"] == "OVERALL" else "  "
        print(f"{marker}{r['condition']:<8}{r['sequence']:<20}{r['n_seeds']:>3}   "
              f"{fmt_ms(r['EPE_mean'], r['EPE_std'], 4):<16}"
              f"{fmt_ms(r['AE_mean'],  r['AE_std'],  2):<16}"
              f"{fmt_ms(r['1PE_mean'], r['1PE_std'], 2):<16}"
              f"{fmt_ms(r['2PE_mean'], r['2PE_std'], 2):<16}"
              f"{fmt_ms(r['3PE_mean'], r['3PE_std'], 2):<16}")
    print(SEP)

    per_seq_condition_csv_path = os.path.join(OUTPUT_DIR, "per_sequence_condition_mean_std.csv")
    with open(per_seq_condition_csv_path, "w", newline="") as f:
        fieldnames = (["condition", "sequence", "n_seeds"] +
                      [f"{k}_{stat}" for k in _METRIC_KEYS for stat in ("mean", "std")] +
                      ["seed_models"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in combined_summary:
            row = {k: r[k] for k in fieldnames if k != "seed_models"}
            row["seed_models"] = "; ".join(r["seed_models"])
            writer.writerow(row)

    print(f"  CSV saved  : {per_seq_condition_csv_path}")

    bin_labels = []
    for name, lo, hi in MAG_BINS:
        bin_labels.append(f"{name} (<{hi:g})" if lo == 0 else
                          f"{name} (>={lo:g})" if hi == float("inf") else
                          f"{name} ({lo:g}-{hi:g})")

    print(f"\n{SEP}")
    print("EPE BY FLOW MAGNITUDE — EACH MODEL  (px, pooled over all test sequences)")
    print(SEP)
    print(f"{'MODEL':<32}" + "".join(f"{lbl:>18}" for lbl in bin_labels))
    print(SEP2)
    for r in model_summaries:
        print(f"{Path(r['model']).name:<32}" + "".join(f"{r[k]:>18.4f}" for k in MAG_KEYS))
    print(SEP2)
    if model_summaries:
        r0 = model_summaries[0]
        shares = [100.0 * r0[f"n_{name}"] / r0["n_pixels"] for name, _, _ in MAG_BINS]
        print(f"{'share of valid GT pixels':<32}" + "".join(f"{s:>17.1f}%" for s in shares))
    print(SEP)

    mag_condition_summary = mean_std_by_condition(model_summaries, group_keys=[],
                                                  metric_keys=MAG_KEYS)
    print(f"\n{SEP}")
    print("EPE BY FLOW MAGNITUDE — MEAN \u00b1 STD ACROSS SEEDS")
    print(SEP)
    print(f"{'CONDITION':<12}{'N':>3}   " + "".join(f"{lbl:<20}" for lbl in bin_labels))
    print(SEP2)
    by_cond = {}
    for r in mag_condition_summary:
        by_cond[r["condition"]] = r
        print(f"{r['condition']:<12}{r['n_seeds']:>3}   " +
              "".join(f"{fmt_ms(r[f'{k}_mean'], r[f'{k}_std'], 4):<20}" for k in MAG_KEYS))
    if "carla" in by_cond and "dsec" in by_cond:
        ratios = [by_cond["carla"][f"{k}_mean"] / by_cond["dsec"][f"{k}_mean"] for k in MAG_KEYS]
        print(f"{'CARLA/DSEC':<12}{'':>3}   " + "".join(f"{x:<20.2f}" for x in ratios))
    print(SEP)

    def _runs(cond):
        return [r for r in model_summaries if infer_condition(r["model"]) == cond]
    mix_runs, dsec_runs = _runs("mix"), _runs("dsec")
    if mix_runs and dsec_runs:
        print("WORST CARLA+DSEC RUN vs BEST DSEC-ONLY RUN  (EPE px; margin > 0 = mix always better)")
        print(SEP2)
        for k, lbl in zip(MAG_KEYS, bin_labels):
            worst_mix = max(r[k] for r in mix_runs)
            best_dsec = min(r[k] for r in dsec_runs)
            print(f"  {lbl:<16} worst mix {worst_mix:.4f}   best dsec {best_dsec:.4f}   "
                  f"margin {best_dsec - worst_mix:+.4f}")
        print(SEP)

    mag_csv_path = os.path.join(OUTPUT_DIR, "magnitude_condition_mean_std.csv")
    with open(mag_csv_path, "w", newline="") as f:
        fieldnames = (["condition", "n_seeds"] +
                      [f"{k}_{stat}" for k in MAG_KEYS for stat in ("mean", "std")] +
                      ["seed_models"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in mag_condition_summary:
            row = {k: r[k] for k in fieldnames if k != "seed_models"}
            row["seed_models"] = "; ".join(r["seed_models"])
            writer.writerow(row)
    print(f"  CSV saved  : {mag_csv_path}")
    print(f"  (per-model magnitude EPEs are also in global_results_all_models.csv)")

    if SAVE_PREDICTIONS:
        print(f"  Flow PNGs  : {OUTPUT_DIR}\\<model>\\<sequence>\\")
    print(SEP)
