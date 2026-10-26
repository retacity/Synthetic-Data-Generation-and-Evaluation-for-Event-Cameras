import numpy as np
import torch
import os
import json
import gc
import cv2
import h5py

CARLA_RUNS_DIR = r"C:\Users\Reta-Fixo\Tese\outputpy"
MODEL_SAVE_DIR = r"C:\Users\Reta-Fixo\Tese\models"
CACHE_DIR      = os.path.join(MODEL_SAVE_DIR, "run_caches")
H_OUT, W_OUT   = 480, 640

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

FLOW_SOURCE = "lidar"

def events_to_voxel(ex, ey, et, ep, H, W, B=5):
    voxel = np.zeros((B, H, W), dtype=np.float32)
    if len(ex) == 0:
        return voxel

    t_min  = et.min()
    t_max  = et.max()
    dt     = t_max - t_min if t_max > t_min else 1.0
    t_norm = np.clip((et - t_min) / dt, 0, 1)

    x   = ex.astype(int)
    y   = ey.astype(int)
    pol = 2 * ep.astype(np.float32) - 1

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, t_norm, pol = x[valid], y[valid], t_norm[valid], pol[valid]

    for b in range(B):
        t_lo = b / B
        t_hi = (b + 1) / B
        mask = (t_norm >= t_lo) & (t_norm < t_hi)
        if mask.sum() == 0:
            continue
        idx      = y[mask] * W + x[mask]
        voxel[b] = np.bincount(idx, weights=pol[mask],
                               minlength=H * W).reshape(H, W)
        m = np.abs(voxel[b]).max()
        if m > 0:
            voxel[b] /= m
    return voxel

INDEX_STRIDE = 100_000

def build_or_load_index(events_path, cache_dir):
    run_name = os.path.basename(os.path.dirname(events_path))
    idx_path = os.path.join(cache_dir, f"{run_name}_tindex.npy")

    if os.path.exists(idx_path):
        return np.load(idx_path)

    print(f"    Building sparse time index (stride={INDEX_STRIDE:,})...")
    with h5py.File(events_path, 'r') as f:
        n       = len(f['t'])
        indices = np.arange(0, n, INDEX_STRIDE)
        t_index = f['t'][indices]

    np.save(idx_path, t_index)
    print(f"    Index: {len(t_index)} entries -> {idx_path}")
    return t_index


def searchsorted_chunked(events_path, t_index, t_query, side='left', chunk_pad=5_000):
    coarse    = int(np.searchsorted(t_index, t_query, side=side))
    coarse    = max(0, coarse - 1)
    global_lo = coarse * INDEX_STRIDE
    global_hi = global_lo + INDEX_STRIDE + chunk_pad

    with h5py.File(events_path, 'r') as f:
        total     = len(f['t'])
        global_hi = min(global_hi, total)
        t_chunk   = f['t'][global_lo:global_hi]

    local_pos = int(np.searchsorted(t_chunk, t_query, side=side))
    return global_lo + local_pos

SHARD_SIZE = 100

def get_shard_path(cache_dir, run_name, shard_idx):
    return os.path.join(cache_dir, f"{run_name}_shard{shard_idx:04d}.pt")

def write_shards(cache_dir, run_name, samples):
    shard_idx = 0
    while os.path.exists(get_shard_path(cache_dir, run_name, shard_idx)):
        shard_idx += 1

    for i in range(0, len(samples), SHARD_SIZE):
        chunk = samples[i : i + SHARD_SIZE]
        torch.save(chunk, get_shard_path(cache_dir, run_name, shard_idx))
        shard_idx += 1

def count_shards(cache_dir, run_name):
    shards = []
    idx    = 0
    while True:
        p = get_shard_path(cache_dir, run_name, idx)
        if not os.path.exists(p):
            break
        shards.append(p)
        idx += 1
    if not shards:
        return [], 0
    total = (len(shards) - 1) * SHARD_SIZE
    last  = torch.load(shards[-1], weights_only=True)
    total += len(last)
    return shards, total

def ler_config_carla(run_dir):
    json_path = os.path.join(run_dir, "config.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            c = json.load(f)
        return {
            "Hz":       c["dvs_hz"],
            "duration": c["duracao_s"],
            "width":    int(c["resolution"].split("x")[0]),
            "height":   int(c["resolution"].split("x")[1]),
        }
    return {"Hz": 1000.0, "duration": 60.0, "width": W_OUT, "height": H_OUT}

def load_flow_source(run_dir, flow_source):
    if flow_source == "lidar":
        flow_path = os.path.join(run_dir, "flow_lidar", "flow.h5")
    else:
        flow_path = os.path.join(run_dir, "flow", "flow.h5")

    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Flow file not found: {flow_path}")

    with h5py.File(flow_path, 'r') as f:
        x_flow_dist = f['x_flow_dist'][:]
        y_flow_dist = f['y_flow_dist'][:]
        timestamps  = f['timestamps'][:]
        valid_mask  = f['valid_mask'][:] if 'valid_mask' in f else None
        t_start_sim = f['t_start_sim'][:] if 't_start_sim' in f else None
        t_end_sim   = f['t_end_sim'][:]   if 't_end_sim'   in f else None

    return x_flow_dist, y_flow_dist, timestamps, valid_mask, t_start_sim, t_end_sim

def processar_run_carla(run_dir, cache_dir, run_name,
                        window_ms=100, step_ms=100, max_amostras=2000,
                        flow_source=FLOW_SOURCE):
    config   = ler_config_carla(run_dir)
    H        = config["height"]
    W        = config["width"]
    DURATION = config["duration"]

    events_path = os.path.join(run_dir, "events.h5")

    if not os.path.exists(events_path):
        print(f"    events.h5 missing -- skipping")
        return 0

    try:
        (x_flow_dist, y_flow_dist, flow_times, valid_mask,
         t_start_sim, t_end_sim) = load_flow_source(run_dir, flow_source)
    except FileNotFoundError as e:
        print(f"    {e} -- skipping")
        return 0

    use_valid_mask = valid_mask is not None

    t_index = build_or_load_index(events_path, cache_dir)

    with h5py.File(events_path, 'r') as f:
        t_min_raw    = int(f['t'][0])
        t_max_raw    = int(f['t'][-1])
        total_events = len(f['t'])
        anchor_t_ns  = f.attrs.get('anchor_event_t_ns', None)
        anchor_sim_s = f.attrs.get('anchor_sim_time_s', None)

    use_intervals = (t_start_sim is not None and t_end_sim is not None
                     and anchor_t_ns is not None)
    if use_intervals:
        def sim_to_ns(t_s):
            return int(anchor_t_ns) + int(round((t_s - float(anchor_sim_s)) * 1e9))
    elif flow_source == "lidar":
        print("    WARNING: old-format LiDAR flow (no t_start_sim/anchor) -- "
              "events/flow will be misaligned by one window. Re-record this run.")

    t_span_ns = t_max_raw - t_min_raw

    print(f"    {total_events/1e6:.1f}M events | index: {len(t_index)} entries | "
          f"span: {t_span_ns/1e9:.2f}s | flow: {flow_source} ({len(flow_times)} frames)")

    window_ns   = window_ms * 1_000_000
    step_ns     = step_ms   * 1_000_000
    if use_intervals:
        num_windows = min(len(t_start_sim), max_amostras)
    else:
        num_windows = min(int(t_span_ns / step_ns), max_amostras)

    n_saved   = 0
    cache_buf = []

    for i in range(num_windows):
        if use_intervals:
            t_lo_ns  = sim_to_ns(t_start_sim[i])
            t_hi_ns  = sim_to_ns(t_end_sim[i])
            flow_idx = i
            if t_hi_ns < t_min_raw or t_lo_ns > t_max_raw:
                continue
        else:
            t_center_ns = t_min_raw + i * step_ns + step_ns // 2
            t_lo_ns     = t_center_ns - window_ns // 2
            t_hi_ns     = t_center_ns + window_ns // 2
            t_center_s = (t_center_ns - t_min_raw) / 1_000_000_000.0
            flow_idx   = int(np.argmin(np.abs(flow_times - t_center_s)))

        idx_lo = searchsorted_chunked(events_path, t_index, t_lo_ns,  side='left')
        idx_hi = searchsorted_chunked(events_path, t_index, t_hi_ns,  side='right')

        if (idx_hi - idx_lo) < 50:
            continue

        with h5py.File(events_path, 'r') as f:
            ex = f['x'][idx_lo:idx_hi]
            ey = f['y'][idx_lo:idx_hi]
            et = f['t'][idx_lo:idx_hi]
            ep = f['pol'][idx_lo:idx_hi]

        voxel = events_to_voxel(ex, ey, et, ep, H, W)
        del ex, ey, et, ep

        flow_u     = x_flow_dist[flow_idx, :, :]
        flow_v     = y_flow_dist[flow_idx, :, :]
        flow_gt    = np.stack([flow_u, flow_v], axis=0)

        mask_frame = valid_mask[flow_idx].astype(bool) if use_valid_mask else np.ones((H, W), dtype=bool)

        sample = (
            torch.tensor(voxel,   dtype=torch.float32),
            torch.tensor(flow_gt, dtype=torch.float32),
            torch.tensor(mask_frame, dtype=torch.bool),
        )


        cache_buf.append(sample)
        n_saved += 1

        if len(cache_buf) >= SHARD_SIZE:
            write_shards(cache_dir, run_name, cache_buf)
            cache_buf.clear()

    if cache_buf:
        write_shards(cache_dir, run_name, cache_buf)
        cache_buf.clear()

    return n_saved


def carregar_todos_runs_carla(runs_dir, cache_dir, flow_source=FLOW_SOURCE):
    os.makedirs(cache_dir, exist_ok=True)

    run_dirs = sorted([
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ])
    print(f"Encontrados {len(run_dirs)} runs CARLA  [flow_source='{flow_source}']")

    index = []

    for i, run_dir in enumerate(run_dirs):
        run_name      = os.path.basename(run_dir)
        shards, total = count_shards(cache_dir, run_name)

        if shards:
            print(f"  [{i+1}/{len(run_dirs)}] {run_name} -- {len(shards)} shards, {total} amostras")
            for p in shards:
                data = torch.load(p, weights_only=True)
                index.append((p, len(data)))
            continue

        print(f"  [{i+1}/{len(run_dirs)}] {run_name}")
        processar_run_carla(run_dir, cache_dir, run_name, flow_source=flow_source)
        shards, total = count_shards(cache_dir, run_name)
        for p in shards:
            data = torch.load(p, weights_only=True)
            index.append((p, len(data)))
        print(f"    -> {total} amostras em {len(shards)} shards")
        gc.collect()

    total_all = sum(n for _, n in index)
    print(f"Total: {total_all} amostras")
    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare CARLA data for training.")
    parser.add_argument(
        "--flow", choices=["camera", "lidar"], default=FLOW_SOURCE,
        help="Flow source to use: 'camera' (dense, default) or 'lidar' (sparse, with valid_mask)."
    )
    args = parser.parse_args()

    print(f"Preparando dados CARLA...  [flow_source='{args.flow}']")
    index = carregar_todos_runs_carla(CARLA_RUNS_DIR, CACHE_DIR, flow_source=args.flow)
    print(f"\nPreparacao concluida! {len(index)} shards prontos em: {CACHE_DIR}")
