import numpy as np
import torch
import os
import gc
import cv2
import h5py
import hdf5plugin
from pathlib import Path
from tqdm import tqdm

DSEC_ROOT  = r"C:\Users\Reta-Fixo\Tese\DSEC_DATASET"
DSEC_CACHE_DIR  = r"C:\Users\Reta-Fixo\Tese\models\dsec_cache"

H_OUT, W_OUT = 480, 640
B_BINS       = 5
WINDOW_MS    = 100

SHARD_SIZE   = 100

os.makedirs(DSEC_CACHE_DIR, exist_ok=True)

INDEX_STRIDE = 100_000

def build_or_load_index(events_h5_path: str, seq_name: str) -> np.ndarray:
    idx_path = os.path.join(DSEC_CACHE_DIR, f"{seq_name}_tindex.npy")
    if os.path.exists(idx_path):
        return np.load(idx_path)

    print(f"    Building sparse time index (stride={INDEX_STRIDE:,}) ...")
    with h5py.File(events_h5_path, 'r') as f:
        n       = len(f['events/t'])
        indices = np.arange(0, n, INDEX_STRIDE)
        t_index = f['events/t'][indices]

    np.save(idx_path, t_index)
    print(f"    Index: {len(t_index)} entries → {idx_path}")
    return t_index


def searchsorted_chunked(events_h5_path: str, t_index: np.ndarray,
                         t_query: int, side: str = 'left',
                         chunk_pad: int = 5_000) -> int:
    coarse    = int(np.searchsorted(t_index, t_query, side=side))
    coarse    = max(0, coarse - 1)
    global_lo = coarse * INDEX_STRIDE
    global_hi = global_lo + INDEX_STRIDE + chunk_pad

    with h5py.File(events_h5_path, 'r') as f:
        total     = len(f['events/t'])
        global_hi = min(global_hi, total)
        t_chunk   = f['events/t'][global_lo:global_hi]

    local_pos = int(np.searchsorted(t_chunk, t_query, side=side))
    return global_lo + local_pos


def load_rectify_map(rectify_h5_path: str) -> np.ndarray:
    with h5py.File(rectify_h5_path, 'r') as f:
        rmap = f['rectify_map'][:]
    return rmap.astype(np.float32)


def apply_rectification(ex: np.ndarray, ey: np.ndarray,
                         et: np.ndarray, ep: np.ndarray,
                         rmap: np.ndarray):
    H, W = rmap.shape[:2]

    in_raw = (ex >= 0) & (ex < W) & (ey >= 0) & (ey < H)
    ex, ey, et, ep = ex[in_raw], ey[in_raw], et[in_raw], ep[in_raw]

    coords = rmap[ey, ex]
    xr     = coords[:, 0]
    yr     = coords[:, 1]

    in_img = (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H)
    return xr[in_img].astype(np.float32), yr[in_img].astype(np.float32), \
           et[in_img], ep[in_img]


def events_to_voxel(ex: np.ndarray, ey: np.ndarray,
                    et: np.ndarray, ep: np.ndarray,
                    H: int, W: int, B: int = 5) -> np.ndarray:
    voxel = np.zeros((B, H, W), dtype=np.float32)
    if len(ex) == 0:
        return voxel

    t_min  = et.min()
    t_max  = et.max()
    dt     = (t_max - t_min) if t_max > t_min else 1.0
    t_norm = np.clip((et - t_min) / dt, 0.0, 1.0)

    x   = ex.astype(np.float32)
    y   = ey.astype(np.float32)
    pol = 2 * ep.astype(np.float32) - 1.0

    valid = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    x, y, t_norm, pol = x[valid], y[valid], t_norm[valid], pol[valid]

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    dx = x - x0
    dy = y - y0
    corners = [
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ]

    for b in range(B):
        t_lo = b / B
        t_hi = (b + 1) / B
        mask = (t_norm >= t_lo) & (t_norm < t_hi)
        if mask.sum() == 0:
            continue
        acc = np.zeros(H * W, dtype=np.float64)
        for ox, oy, w in corners:
            xi = x0[mask] + ox
            yi = y0[mask] + oy
            wi = w[mask] * pol[mask]
            ok = (xi < W) & (yi < H) & (wi != 0)
            acc += np.bincount(yi[ok] * W + xi[ok], weights=wi[ok],
                               minlength=H * W)
        voxel[b] = acc.reshape(H, W).astype(np.float32)
        m = np.abs(voxel[b]).max()
        if m > 0:
            voxel[b] /= m
    return voxel


def load_flow_png(png_path) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot open flow PNG: {png_path}")

    flow_x = (img[:, :, 2].astype(np.float32) - 2**15) / 128.0
    flow_y = (img[:, :, 1].astype(np.float32) - 2**15) / 128.0
    valid  =  img[:, :, 0].astype(bool)

    return np.stack([flow_x, flow_y], axis=0), valid


def get_shard_path(seq_name: str, shard_idx: int) -> str:
    return os.path.join(DSEC_CACHE_DIR, f"{seq_name}_shard{shard_idx:04d}.pt")


def count_existing_shards(seq_name: str) -> int:
    idx = 0
    while os.path.exists(get_shard_path(seq_name, idx)):
        idx += 1
    return idx


def flush_shard(seq_name: str, buf: list, shard_idx: int):
    torch.save(buf, get_shard_path(seq_name, shard_idx))


def prepare_sequence(seq_dir: str) -> int:
    seq_name   = os.path.basename(seq_dir)
    events_h5  = os.path.join(seq_dir, "events.h5")
    rectify_h5 = os.path.join(seq_dir, "rectify_map.h5")
    flow_dir   = os.path.join(seq_dir, "flow")
    ts_file    = os.path.join(seq_dir, "forward_timestamps.txt")

    missing = [p for p in [events_h5, rectify_h5, flow_dir, ts_file]
               if not os.path.exists(p)]
    if missing:
        print(f"  [{seq_name}] SKIP — missing files: {missing}")
        return 0

    if count_existing_shards(seq_name) > 0:
        n_shards = count_existing_shards(seq_name)
        last     = torch.load(get_shard_path(seq_name, n_shards - 1),
                              weights_only=True)
        total    = (n_shards - 1) * SHARD_SIZE + len(last)
        print(f"  [{seq_name}] Already cached — {n_shards} shards, {total} samples")
        return 0

    print(f"\n{'='*60}")
    print(f"  Preparing: {seq_name}")

    rmap      = load_rectify_map(rectify_h5)
    H_ev, W_ev = rmap.shape[:2]

    flow_pngs = sorted(Path(flow_dir).glob("*.png"))

    raw_ts = np.loadtxt(ts_file, delimiter=',', dtype=np.int64, skiprows=1)
    if raw_ts.ndim == 2:
        flow_ts_start = raw_ts[:, 0]
        flow_ts_end   = raw_ts[:, 1]
    else:
        flow_ts_start = raw_ts
        flow_ts_end   = raw_ts

    if len(flow_ts_start) != len(flow_pngs):
        n = min(len(flow_ts_start), len(flow_pngs))
        print(f"  [{seq_name}] WARNING: timestamp/PNG count mismatch "
              f"({len(flow_ts_start)} vs {len(flow_pngs)}) — using {n}")
        flow_ts_start = flow_ts_start[:n]
        flow_ts_end   = flow_ts_end[:n]
        flow_pngs     = flow_pngs[:n]

    t_index = build_or_load_index(events_h5, seq_name)


    with h5py.File(events_h5, 'r') as f:
        t_min_ev = int(f['events/t'][0])
        t_max_ev = int(f['events/t'][-1])
        total_ev = len(f['events/t'])
        t_offset = int(f['t_offset'][()])

    flow_ts_start = flow_ts_start.astype(np.int64) - np.int64(t_offset)
    flow_ts_end   = flow_ts_end.astype(np.int64)   - np.int64(t_offset)

    print(f"    {total_ev/1e6:.1f}M events | "
          f"{len(flow_pngs)} flow frames | "
          f"res {W_ev}×{H_ev}")

    shard_buf = []
    shard_idx = 0
    n_saved   = 0
    n_skipped = 0

    for (t_lo_us, t_hi_us), png_path in tqdm(
            zip(zip(flow_ts_start, flow_ts_end), flow_pngs),
            total=len(flow_pngs), desc=f"  {seq_name}", ncols=80):
        t_lo_us = int(t_lo_us)
        t_hi_us = int(t_hi_us)

        if t_hi_us < t_min_ev or t_lo_us > t_max_ev:
            n_skipped += 1
            continue
        t_lo_us = max(t_lo_us, t_min_ev)
        t_hi_us = min(t_hi_us, t_max_ev)

        idx_lo = searchsorted_chunked(events_h5, t_index, t_lo_us, side='left')
        idx_hi = searchsorted_chunked(events_h5, t_index, t_hi_us, side='right')

        if (idx_hi - idx_lo) < 50:
            n_skipped += 1
            continue

        with h5py.File(events_h5, 'r') as f:
            ex = f['events/x'][idx_lo:idx_hi]
            ey = f['events/y'][idx_lo:idx_hi]
            et = f['events/t'][idx_lo:idx_hi]
            ep = f['events/p'][idx_lo:idx_hi]

        xr, yr, et_r, ep_r = apply_rectification(ex, ey, et, ep, rmap)
        del ex, ey, et, ep

        voxel = events_to_voxel(xr, yr, et_r, ep_r, H_ev, W_ev, B=B_BINS)
        del xr, yr, et_r, ep_r

        flow_gt, valid_mask = load_flow_png(png_path)

        if H_ev != H_OUT or W_ev != W_OUT:
            scale_x = W_OUT / W_ev
            scale_y = H_OUT / H_ev
            voxel = np.stack([
                cv2.resize(voxel[c], (W_OUT, H_OUT), interpolation=cv2.INTER_LINEAR)
                for c in range(B_BINS)
            ])

            flow_gt = np.stack([
                cv2.resize(flow_gt[0], (W_OUT, H_OUT), interpolation=cv2.INTER_NEAREST) * scale_x,
                cv2.resize(flow_gt[1], (W_OUT, H_OUT), interpolation=cv2.INTER_NEAREST) * scale_y,
            ])
            valid_mask = cv2.resize(
                valid_mask.astype(np.uint8), (W_OUT, H_OUT),
                interpolation=cv2.INTER_NEAREST).astype(bool)

        shard_buf.append((
            torch.tensor(voxel,      dtype=torch.float32),
            torch.tensor(flow_gt,    dtype=torch.float32),
            torch.tensor(valid_mask, dtype=torch.bool),
        ))
        n_saved += 1

        if len(shard_buf) >= SHARD_SIZE:
            flush_shard(seq_name, shard_buf, shard_idx)
            shard_idx += 1
            shard_buf.clear()
            gc.collect()

    if shard_buf:
        flush_shard(seq_name, shard_buf, shard_idx)
        shard_buf.clear()

    print(f"    → {n_saved} samples in {shard_idx + 1} shards "
          f"({n_skipped} frames skipped)")
    return n_saved

if __name__ == "__main__":
    seq_dirs = sorted([
        os.path.join(DSEC_ROOT, d)
        for d in os.listdir(DSEC_ROOT)
        if os.path.isdir(os.path.join(DSEC_ROOT, d))
    ])
    print(f"Found {len(seq_dirs)} sequences in {DSEC_ROOT}")
    print(f"Cache dir: {DSEC_CACHE_DIR}\n")

    total_samples = 0
    for seq_dir in seq_dirs:
        n = prepare_sequence(seq_dir)
        total_samples += n
        gc.collect()

    print(f"\n{'='*60}")
    print(f"Preparation complete.  {total_samples} new samples cached.")
    print(f"Cache: {DSEC_CACHE_DIR}")
