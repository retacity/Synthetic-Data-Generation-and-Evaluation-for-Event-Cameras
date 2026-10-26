# Event-Based Optical Flow — CARLA + DSEC

Code for generating synthetic event-camera data in CARLA, preparing it together with the real DSEC dataset, and training/evaluating a U-Net that predicts optical flow from event voxel grids.

## File organisation

| File | Purpose |
|---|---|
| `carla_dvs_flow_runs.py` | Records synthetic data in CARLA 0.9.15: DVS events, grayscale frames, CARLA optical-flow camera, and LiDAR-based sparse flow (DSEC-style ground truth). Runs are defined in `runs_config.json`. |
| `prepare_carla.py` | Converts the recorded CARLA runs into 5-bin voxel grids + flow ground truth, saved as sharded `.pt` cache files. |
| `prepare_dsec.py` | Converts DSEC sequences (rectified events + flow PNGs) into the same sharded `.pt` cache format. |
| `train_unet3x3.py` | Interactive training script. Trains the U-Net on CARLA only, DSEC only, or CARLA + DSEC. |
| `test_unet3x3_all.py` | Evaluates trained models on a fixed set of DSEC sequences and writes EPE / AE / NPE metrics to CSV, grouped by condition and seed. |

Note: `train_unet3x3.py` imports its paths from `prepare_carla.py` and `prepare_dsec.py`, so changing them there is enough.

## Pipeline

Run the scripts in this order:

```
1. carla_dvs_flow_runs.py      record CARLA runs        -> <BASE_OUTPUT_DIR>/<run>/
2. prepare_carla.py            cache CARLA runs         -> <MODEL_SAVE_DIR>/run_caches/
3. prepare_dsec.py             cache DSEC sequences     -> <DSEC_CACHE_DIR>/
4. train_unet3x3.py            train models             -> <MODEL_SAVE_DIR>/<run_name>_best.pt
5. test_unet3x3_all.py         evaluate on DSEC         -> <OUTPUT_DIR>/*.csv
```

`prepare_carla.py` takes `--flow lidar` (default) or `--flow camera` to choose the ground-truth source.

For multiple seeds, the training seed can be set with an environment variable, e.g. `TRAIN_SEED=1 python train_unet3x3.py`.

## ⚠️ Paths must be changed

All paths are hard-coded to the original machine (`C:\Users\Reta-Fixo\Tese\...`). Before running anything, edit the constants at the top of each script to match your own folders:

| File | Constants to change |
|---|---|
| `carla_dvs_flow_runs.py` | `CARLA_PATH` (CarlaUE4.exe), `BASE_OUTPUT_DIR` (where runs are recorded), `RUNS_CONFIG` (path to `runs_config.json`) |
| `prepare_carla.py` | `CARLA_RUNS_DIR` (same folder as `BASE_OUTPUT_DIR`), `MODEL_SAVE_DIR` (models + CARLA cache) |
| `prepare_dsec.py` | `DSEC_ROOT` (DSEC sequences), `DSEC_CACHE_DIR` (DSEC cache output) |
| `test_unet3x3_all.py` | `CACHE_DIR` (same as `DSEC_CACHE_DIR`), `MODELS_DIR`, `OUTPUT_DIR`, `EXTRA_MODEL_DIRS` |

Also check in `test_unet3x3_all.py`:
- `MODELS_TO_TEST` — checkpoint filenames to evaluate (set to `[]` to pick interactively).
- `SEQUENCES_TO_TEST` — DSEC sequences used for evaluation.
- `CONDITION_OVERRIDES` — maps checkpoint names that don't contain `carla` / `dsec` / `mix` to a condition.

## Expected DSEC layout

`prepare_dsec.py` expects one folder per sequence inside `DSEC_ROOT`:

```
DSEC_ROOT/
  <sequence>/
    events.h5                 events/x, events/y, events/t, events/p, t_offset
    rectify_map.h5
    forward_timestamps.txt
    flow/
      000000.png ...          16-bit DSEC flow PNGs
```

## runs_config.json

A list of runs for `carla_dvs_flow_runs.py`. If the file is missing, the script prints an example. Minimal entry:

```json
[
  {"map": "Town01", "weather": "ClearNoon", "spawn_idx": 0, "npc_vehicles": 30, "npc_pedestrians": 20}
]
```

Valid maps: `Town01`–`Town07`, `Town10HD`. Valid weather: `ClearNoon`, `CloudyNoon`, `ClearSunset`, `ClearNight`.

## Requirements

- CARLA 0.9.15 (server + matching `carla` Python package)
- Python packages: `numpy`, `torch`, `opencv-python`, `h5py`, `hdf5plugin`, `tqdm`
- A CUDA GPU is recommended for training and evaluation.
