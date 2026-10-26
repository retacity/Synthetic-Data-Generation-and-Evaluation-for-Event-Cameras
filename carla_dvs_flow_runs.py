from __future__ import annotations
import carla
import os
import random
import numpy as np
import time
import datetime
import cv2
import subprocess
import sys
import threading
import h5py
import json

CARLA_PATH    = r"C:\Users\Reta-Fixo\Tese\Carla_0.9.15\WindowsNoEditor\CarlaUE4.exe"
CARLA_PORT    = 2000
BASE_OUTPUT_DIR = r"C:\Users\Reta-Fixo\Tese\outputpy"
RUNS_CONFIG   = r"C:\Users\Reta-Fixo\Tese\src\runs_config.json"

Hz           = 10.0
DURATION     = 30.0
WIDTH        = 640
HEIGHT       = 480
Hz_Gray_Flow = 10.0
EGO_BP       = "vehicle.tesla.model3"

DVS_POSITIVE_THRESHOLD       = 0.175
DVS_NEGATIVE_THRESHOLD       = 0.175
DVS_SIGMA_POSITIVE_THRESHOLD = 0.01
DVS_SIGMA_NEGATIVE_THRESHOLD = 0.01
DVS_REFRACTORY_PERIOD_NS     = 0
DVS_USE_LOG                  = True
DVS_LOG_EPS                  = 0.001
FOV = 60.0
CAMERA_PITCH = 0.0

LIDAR_CHANNELS       = 64
LIDAR_RANGE          = 120.0
LIDAR_ROTATION_FREQ  = 10.0
LIDAR_POINTS_PER_SEC = 2600000
LIDAR_UPPER_FOV      = 12.0
LIDAR_LOWER_FOV      = -24.0
LIDAR_DROPOFF_RATE   = 0.0
LIDAR_NOISE_STDDEV   = 0.0

DYNAMIC_SEMANTIC_TAGS = frozenset({12, 13, 14, 15, 16, 17, 18, 19})

TM_PORT = 8000

NUM_NPC_VEHICLES    = 10
NUM_NPC_PEDESTRIANS = 10

EGO_SPEED_DIFF_PCT             = -150.0
EGO_LANE_OFFSET                = 0.0
EGO_GLOBAL_DISTANCE_TO_LEADING = 2.0

NPC_GLOBAL_DISTANCE_TO_LEADING = 2.5
NPC_SPEED_DIFF_MIN             = -30.0
NPC_SPEED_DIFF_MAX             = -10.0
NPC_LANECHANGE_PCT             = 10.0
NPC_VEHICLE_BLUEPRINTS         = []

PEDESTRIAN_SPEED_MIN = 1.0
PEDESTRIAN_SPEED_MAX = 2.0

AVAILABLE_MAPS = [
    "Town01", "Town02", "Town03", "Town04",
    "Town05", "Town06", "Town07", "Town10HD",
]

WEATHER_PRESETS = {
    "ClearNoon":   carla.WeatherParameters.ClearNoon,
    "CloudyNoon":  carla.WeatherParameters.CloudyNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "ClearNight":  carla.WeatherParameters.ClearNight,
}

def launch_carla():
    print("Launching CARLA server...")
    subprocess.Popen([
        CARLA_PATH,
        "-carla-server",
        "-benchmark",
        "-windowed",
    ], creationflags=subprocess.CREATE_NO_WINDOW)
    print("Waiting for CARLA to initialize...")
    max_wait = 60
    for i in range(max_wait):
        try:
            client = carla.Client("localhost", CARLA_PORT)
            client.set_timeout(60.0)
            client.get_server_version()
            print("CARLA ready!")
            return client
        except:
            print(f"Waiting {i+1}/{max_wait}s...", end="\r")
            time.sleep(1)
    print("\nERROR: CARLA failed to start!")
    sys.exit(1)

def connect_carla():
    try:
        client = carla.Client("localhost", CARLA_PORT)
        client.set_timeout(10.0)
        client.get_server_version()
        print("CARLA is already running!")
    except:
        client = launch_carla()
    client.set_timeout(60.0)
    return client

def set_synchronous_mode(world, hz):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / hz
    world.apply_settings(settings)

def restore_async_mode(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

def get_npc_vehicle_blueprints(world, patterns):
    library = world.get_blueprint_library()
    if not patterns:
        return list(library.filter("vehicle.*"))
    blueprints = {}
    for pattern in patterns:
        matches = library.filter(pattern)
        if not matches:
            print(f"  WARNING: no vehicle blueprint matched '{pattern}' — skipping")
        for bp in matches:
            blueprints[bp.id] = bp
    if not blueprints:
        raise ValueError(f"No vehicle blueprints matched npc_vehicle_blueprints: {patterns}")
    return list(blueprints.values())


def spawn_npc_vehicles(world, client, num_vehicles, p):
    if num_vehicles == 0:
        return []
    tm_port             = p["tm_port"]
    vehicle_blueprints = get_npc_vehicle_blueprints(world, p["npc_vehicle_blueprints"])
    spawn_points       = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(p["npc_global_distance_to_leading_vehicle"])
    traffic_manager.set_respawn_dormant_vehicles(True)

    npcs = []
    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        try:
            actor = world.try_spawn_actor(bp, spawn_points[i])
            if actor:
                actor.set_autopilot(True, tm_port)
                traffic_manager.vehicle_percentage_speed_difference(
                    actor, random.uniform(p["npc_speed_diff_min"], p["npc_speed_diff_max"]))
                traffic_manager.random_left_lanechange_percentage(actor, p["npc_lanechange_pct"])
                traffic_manager.random_right_lanechange_percentage(actor, p["npc_lanechange_pct"])
                npcs.append(actor)
        except:
            pass
    print(f"  {len(npcs)}/{num_vehicles} NPC vehicles spawned")
    return npcs


def spawn_npc_pedestrians(world, client, num_pedestrians, p):
    if num_pedestrians == 0:
        return [], []
    blueprint_library    = world.get_blueprint_library()
    walker_blueprints    = blueprint_library.filter("walker.pedestrian.*")
    walker_controller_bp = blueprint_library.find("controller.ai.walker")

    spawn_points = []
    for _ in range(num_pedestrians):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))

    walkers, controllers = [], []
    batch   = [carla.command.SpawnActor(random.choice(walker_blueprints), sp) for sp in spawn_points]
    results = client.apply_batch_sync(batch, True)

    walker_ids = []
    for r in results:
        if not r.error:
            walkers.append(world.get_actor(r.actor_id))
            walker_ids.append(r.actor_id)

    batch_ctrl   = [carla.command.SpawnActor(walker_controller_bp, carla.Transform(), wid) for wid in walker_ids]
    results_ctrl = client.apply_batch_sync(batch_ctrl, True)
    for r in results_ctrl:
        if not r.error:
            controllers.append(world.get_actor(r.actor_id))

    world.tick()
    for ctrl in controllers:
        if ctrl:
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
            ctrl.set_max_speed(random.uniform(p["pedestrian_speed_min"], p["pedestrian_speed_max"]))

    print(f"  {len(walkers)}/{num_pedestrians} NPC pedestrians spawned")
    return walkers, controllers

def spawn_vehicle(world, client, blueprint_id, spawn_point, p):
    bp = world.get_blueprint_library().find(blueprint_id)
    vehicle = world.try_spawn_actor(bp, spawn_point)

    if vehicle is None:
        print("WARNING: Spawn point blocked! Finding a new one...")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for pt in spawn_points:
            vehicle = world.try_spawn_actor(bp, pt)
            if vehicle is not None:
                break

    if vehicle:
        tm_port = p["tm_port"]
        vehicle.set_autopilot(True, tm_port)
        tm = client.get_trafficmanager(tm_port)
        tm.vehicle_percentage_speed_difference(vehicle, p["ego_speed_diff_pct"])
        tm.vehicle_lane_offset(vehicle, p["ego_lane_offset"])
        tm.set_global_distance_to_leading_vehicle(p["ego_global_distance_to_leading_vehicle"])
        return vehicle
    else:
        raise RuntimeError("Could not find any free spawn point for Ego.")

def spawn_dvs_camera(world, vehicle, hz, p):
    bp = world.get_blueprint_library().find("sensor.camera.dvs")
    w, h = map(int, p["resolution"].split("x"))
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("sensor_tick",  str(1.0 / hz))
    bp.set_attribute("positive_threshold",       str(p["dvs_positive_threshold"]))
    bp.set_attribute("negative_threshold",       str(p["dvs_negative_threshold"]))
    bp.set_attribute("sigma_positive_threshold", str(p["dvs_sigma_positive_threshold"]))
    bp.set_attribute("sigma_negative_threshold", str(p["dvs_sigma_negative_threshold"]))
    bp.set_attribute("refractory_period_ns",     str(p["dvs_refractory_period_ns"]))
    bp.set_attribute("use_log",                  str(p["dvs_use_log"]).lower())
    bp.set_attribute("log_eps",                  str(p["dvs_log_eps"]))
    bp.set_attribute("fov",                      str(p["field_of_view"]))
    return world.spawn_actor(bp,
        carla.Transform(carla.Location(x=0.0, z=2.4), carla.Rotation(pitch=p["camera_pitch"])),
        attach_to=vehicle)

def spawn_grayscale_camera(world, vehicle, p):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    w, h = map(int, p["resolution"].split("x"))
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("sensor_tick",  str(1.0 / p["gray_flow_fps"]))
    bp.set_attribute("fov",          str(p["field_of_view"]))
    return world.spawn_actor(bp,
        carla.Transform(carla.Location(x=0.0, z=2.4), carla.Rotation(pitch=p["camera_pitch"])),
        attach_to=vehicle)

def spawn_flow_camera(world, vehicle, p):
    bp = world.get_blueprint_library().find("sensor.camera.optical_flow")
    w, h = map(int, p["resolution"].split("x"))
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("sensor_tick",  str(1.0 / p["gray_flow_fps"]))
    bp.set_attribute("fov",          str(p["field_of_view"]))
    return world.spawn_actor(bp,
        carla.Transform(carla.Location(x=0.0, z=2.4), carla.Rotation(pitch=p["camera_pitch"])),
        attach_to=vehicle)

def spawn_lidar(world, vehicle, hz, p):
    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
    bp.set_attribute("channels",           str(p["lidar_channels"]))
    bp.set_attribute("range",              str(p["lidar_range_m"]))
    bp.set_attribute("rotation_frequency", str(p["lidar_rotation_hz"]))
    bp.set_attribute("points_per_second",  str(p["lidar_points_per_sec"]))
    bp.set_attribute("upper_fov",          str(p["lidar_upper_fov"]))
    bp.set_attribute("lower_fov",          str(p["lidar_lower_fov"]))
    bp.set_attribute("sensor_tick",        str(1.0 / hz))
    return world.spawn_actor(bp,
        carla.Transform(carla.Location(x=0.0, z=2.4), carla.Rotation(pitch=p["camera_pitch"])),
        attach_to=vehicle)

class LidarFlowEstimator:

    _K: np.ndarray = None

    @classmethod
    def _get_K(cls) -> np.ndarray:
        if cls._K is None:
            fx = fy = (WIDTH / 2.0) / np.tan(np.deg2rad(FOV / 2.0))
            cx, cy  = WIDTH / 2.0, HEIGHT / 2.0
            cls._K  = np.array(
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0,  0,  1]],
                dtype=np.float64,
            )
        return cls._K

    def __init__(self, sim_hz: float, flow_fps: float):
        self._ticks_per_frame = max(1, round(sim_hz / flow_fps))
        self._tick_count      = 0
        self._accum_world: list[np.ndarray] = []
        self._prev_world      = None
        self._prev_transform  = None
        self._prev_time       = None
        self.flow_frames:  list[np.ndarray] = []
        self.valid_frames: list[np.ndarray] = []
        self.frame_t0: list[float] = []
        self.frame_t1: list[float] = []

        self._total_points_seen     = 0
        self._dynamic_points_removed = 0

    _SEMANTIC_LIDAR_DTYPE = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32),
        ('ObjIdx', np.uint32), ('ObjTag', np.uint32),
    ])

    def _parse_lidar(self, raw_data) -> np.ndarray:
        pts = np.frombuffer(raw_data.raw_data, dtype=self._SEMANTIC_LIDAR_DTYPE)

        self._total_points_seen += len(pts)
        static_mask = ~np.isin(pts['ObjTag'], list(DYNAMIC_SEMANTIC_TAGS))
        self._dynamic_points_removed += int((~static_mask).sum())

        xyz = np.stack(
            [pts['x'][static_mask], pts['y'][static_mask], pts['z'][static_mask]],
            axis=1,
        ).astype(np.float32)
        return xyz

    @staticmethod
    def _to_camera_frame(xyz: np.ndarray) -> np.ndarray:
        cam = np.empty_like(xyz)
        cam[:, 0] =  xyz[:, 1]
        cam[:, 1] = -xyz[:, 2]
        cam[:, 2] =  xyz[:, 0]
        return cam

    @staticmethod
    def _project(xyz_cam: np.ndarray, K: np.ndarray):
        depth = xyz_cam[:, 2]
        valid = (depth > 0.1) & \
                (xyz_cam[:, 0] / depth * K[0, 0] + K[0, 2] >= 0) & \
                (xyz_cam[:, 0] / depth * K[0, 0] + K[0, 2] <  WIDTH) & \
                (xyz_cam[:, 1] / depth * K[1, 1] + K[1, 2] >= 0) & \
                (xyz_cam[:, 1] / depth * K[1, 1] + K[1, 2] <  HEIGHT)
        pts  = xyz_cam[valid]
        u    = K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2]
        v    = K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]
        return u, v, valid

    @staticmethod
    def _sensor_to_world(xyz: np.ndarray, T: carla.Transform) -> np.ndarray:
        M   = np.array(T.get_matrix(), dtype=np.float64)
        hom = np.hstack([xyz.astype(np.float64), np.ones((len(xyz), 1))])
        return (M @ hom.T).T[:, :3]

    @staticmethod
    def _world_to_sensor(xyz_w: np.ndarray, T: carla.Transform) -> np.ndarray:
        M_inv = np.array(T.get_inverse_matrix(), dtype=np.float64)
        hom   = np.hstack([xyz_w, np.ones((len(xyz_w), 1))])
        return (M_inv @ hom.T).T[:, :3]

    def _estimate_flow(
        self,
        pts_world: np.ndarray,
        T0:        carla.Transform,
        T1:        carla.Transform,
    ) -> tuple[np.ndarray, np.ndarray]:
        K = self._get_K()
        flow  = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
        valid = np.zeros((HEIGHT, WIDTH),    dtype=bool)
        if len(pts_world) == 0:
            return flow, valid

        cam0           = self._to_camera_frame(self._world_to_sensor(pts_world, T0))
        u0, v0, valid0 = self._project(cam0, K)
        idx0           = valid0.nonzero()[0]
        depth0         = cam0[idx0, 2]

        cam1           = self._to_camera_frame(self._world_to_sensor(pts_world[idx0], T1))
        u1, v1, valid1 = self._project(cam1, K)
        idx1           = valid1.nonzero()[0]
        if len(idx1) == 0:
            return flow, valid

        u0s, v0s, d0s = u0[idx1], v0[idx1], depth0[idx1]
        du = (u1 - u0s).astype(np.float32)
        dv = (v1 - v0s).astype(np.float32)

        u0_px = u0s.astype(np.int64).clip(0, WIDTH  - 1)
        v0_px = v0s.astype(np.int64).clip(0, HEIGHT - 1)

        pix   = v0_px * WIDTH + u0_px
        order = np.lexsort((d0s, pix))
        pix_sorted = pix[order]
        _, first   = np.unique(pix_sorted, return_index=True)
        keep       = order[first]

        flow[v0_px[keep], u0_px[keep], 0] = du[keep]
        flow[v0_px[keep], u0_px[keep], 1] = dv[keep]
        valid[v0_px[keep], u0_px[keep]]   = True
        return flow, valid

    def callback(self, raw_data):
        pts_cur = self._parse_lidar(raw_data)
        T_cur   = raw_data.transform
        t_cur   = float(raw_data.timestamp)

        self._tick_count += 1

        if len(pts_cur):
            self._accum_world.append(self._sensor_to_world(pts_cur, T_cur))

        if self._tick_count >= self._ticks_per_frame:
            full_world = (np.concatenate(self._accum_world, axis=0)
                          if self._accum_world else np.empty((0, 3)))
            if self._prev_world is not None:
                flow, valid = self._estimate_flow(self._prev_world,
                                                  self._prev_transform, T_cur)
                self.flow_frames.append(flow)
                self.valid_frames.append(valid)
                self.frame_t0.append(self._prev_time)
                self.frame_t1.append(t_cur)
            self._prev_world     = full_world
            self._prev_transform = T_cur
            self._prev_time      = t_cur
            self._tick_count     = 0
            self._accum_world    = []

def follow_camera(camera, spectator, stop_event):
    while not stop_event.is_set():
        try:
            spectator.set_transform(camera.get_transform())
        except:
            pass
        time.sleep(0.05)

def cleanup(actors_to_destroy):
    print("Destroying actors...")
    for actor in actors_to_destroy:
        try:
            if actor and actor.is_alive:
                if hasattr(actor, 'stop'):
                    try:
                        actor.stop()
                    except:
                        pass
                actor.destroy()
        except:
            pass

def force_green_light(vehicle, world):
    waypoint = world.get_map().get_waypoint(vehicle.get_location())
    traffic_lights = world.get_traffic_lights_from_waypoint(waypoint, 30.0)
    for light in traffic_lights:
        if light.get_state() != carla.TrafficLightState.Green:
            light.set_state(carla.TrafficLightState.Green)
            light.set_green_time(20.0)

def run_single(client, cfg, run_number, total_runs):
    p = {
        "map":                          cfg["map"],
        "weather":                      cfg["weather"],
        "spawn_idx":                    int(cfg.get("spawn_idx",                0)),
        "num_npc_vehicles":             int(cfg.get("npc_vehicles",             NUM_NPC_VEHICLES)),
        "num_npc_pedestrians":          int(cfg.get("npc_pedestrians",          NUM_NPC_PEDESTRIANS)),
        "dvs_hz":                       float(cfg.get("dvs_hz",                 Hz)),
        "tm_port":                      int(cfg.get("tm_port",                  TM_PORT)),
        "resolution":                   cfg.get("resolution",                   f"{WIDTH}x{HEIGHT}"),
        "gray_flow_fps":                float(cfg.get("gray_flow_fps",          Hz_Gray_Flow)),
        "ego_vehicle":                  cfg.get("ego_vehicle",                  EGO_BP),
        "duracao_s":                    float(cfg.get("duracao_s",              DURATION)),
        "dvs_positive_threshold":       float(cfg.get("dvs_positive_threshold",       DVS_POSITIVE_THRESHOLD)),
        "dvs_negative_threshold":       float(cfg.get("dvs_negative_threshold",       DVS_NEGATIVE_THRESHOLD)),
        "dvs_sigma_positive_threshold": float(cfg.get("dvs_sigma_positive_threshold", DVS_SIGMA_POSITIVE_THRESHOLD)),
        "dvs_sigma_negative_threshold": float(cfg.get("dvs_sigma_negative_threshold", DVS_SIGMA_NEGATIVE_THRESHOLD)),
        "dvs_refractory_period_ns":     int(cfg.get("dvs_refractory_period_ns",       DVS_REFRACTORY_PERIOD_NS)),
        "dvs_use_log":                  bool(cfg.get("dvs_use_log",                   DVS_USE_LOG)),
        "dvs_log_eps":                  float(cfg.get("dvs_log_eps",                  DVS_LOG_EPS)),
        "field_of_view":                float(cfg.get("field_of_view",          FOV)),
        "camera_pitch":                 float(cfg.get("camera_pitch",           CAMERA_PITCH)),
        "lidar_channels":               int(cfg.get("lidar_channels",           LIDAR_CHANNELS)),
        "lidar_range_m":                float(cfg.get("lidar_range_m",          LIDAR_RANGE)),
        "lidar_rotation_hz":            float(cfg.get("lidar_rotation_hz",      LIDAR_ROTATION_FREQ)),
        "lidar_points_per_sec":         int(cfg.get("lidar_points_per_sec",     LIDAR_POINTS_PER_SEC)),
        "lidar_upper_fov":              float(cfg.get("lidar_upper_fov",        LIDAR_UPPER_FOV)),
        "lidar_lower_fov":              float(cfg.get("lidar_lower_fov",        LIDAR_LOWER_FOV)),
        "lidar_dropoff_rate":           float(cfg.get("lidar_dropoff_rate",     LIDAR_DROPOFF_RATE)),
        "lidar_noise_stddev":           float(cfg.get("lidar_noise_stddev",     LIDAR_NOISE_STDDEV)),
        "ego_speed_diff_pct":                    float(cfg.get("ego_speed_diff_pct",                    EGO_SPEED_DIFF_PCT)),
        "ego_lane_offset":                       float(cfg.get("ego_lane_offset",                       EGO_LANE_OFFSET)),
        "ego_global_distance_to_leading_vehicle": float(cfg.get("ego_global_distance_to_leading_vehicle", EGO_GLOBAL_DISTANCE_TO_LEADING)),
        "npc_global_distance_to_leading_vehicle": float(cfg.get("npc_global_distance_to_leading_vehicle", NPC_GLOBAL_DISTANCE_TO_LEADING)),
        "npc_speed_diff_min":                    float(cfg.get("npc_speed_diff_min",                    NPC_SPEED_DIFF_MIN)),
        "npc_speed_diff_max":                    float(cfg.get("npc_speed_diff_max",                    NPC_SPEED_DIFF_MAX)),
        "npc_lanechange_pct":                    float(cfg.get("npc_lanechange_pct",                    NPC_LANECHANGE_PCT)),
        "npc_vehicle_blueprints":                list(cfg.get("npc_vehicle_blueprints",                 NPC_VEHICLE_BLUEPRINTS)),
        "pedestrian_speed_min":                  float(cfg.get("pedestrian_speed_min",                  PEDESTRIAN_SPEED_MIN)),
        "pedestrian_speed_max":                  float(cfg.get("pedestrian_speed_max",                  PEDESTRIAN_SPEED_MAX)),
    }

    map_name             = p["map"]
    weather_name          = p["weather"]
    spawn_idx             = p["spawn_idx"]
    num_npc_vehicles      = p["num_npc_vehicles"]
    num_npc_pedestrians   = p["num_npc_pedestrians"]
    hz                    = p["dvs_hz"]
    W, H = map(int, p["resolution"].split("x"))
    DUR  = p["duracao_s"]
    EGO  = p["ego_vehicle"]

    assert p["lidar_rotation_hz"] == p["gray_flow_fps"], (
        f"lidar_rotation_hz ({p['lidar_rotation_hz']}) must equal gray_flow_fps "
        f"({p['gray_flow_fps']}) — see LIDAR_ROTATION_FREQ / Hz_Gray_Flow constants."
    )

    print(f"\n{'='*55}")
    print(f"  RUN {run_number}/{total_runs}: {map_name} | {weather_name} | spawn={spawn_idx} | {num_npc_vehicles} cars | {num_npc_pedestrians} pedestrians")
    print(f"{'='*55}")

    print(f"Loading map {map_name}...")
    world = client.load_world(map_name)
    time.sleep(2.0)
    world.set_weather(WEATHER_PRESETS[weather_name])
    set_synchronous_mode(world, hz)
    print(f"  Map: {world.get_map().name} | Weather: {weather_name} | Sync: {hz}Hz")

    run_id        = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    auto_label    = f"{run_id}_{map_name}_{weather_name}_v{num_npc_vehicles}_p{num_npc_pedestrians}"
    run_label     = cfg.get("name", auto_label)
    output_dir    = os.path.join(BASE_OUTPUT_DIR, run_label)
    grayscale_dir = os.path.join(output_dir, "grayscale")
    flow_dir      = os.path.join(output_dir, "flow")
    lidar_flow_dir = os.path.join(output_dir, "flow_lidar")
    os.makedirs(output_dir,     exist_ok=True)
    os.makedirs(grayscale_dir,  exist_ok=True)
    os.makedirs(flow_dir,       exist_ok=True)
    os.makedirs(lidar_flow_dir, exist_ok=True)

    print("\nSpawning traffic...")
    npc_vehicles_list            = spawn_npc_vehicles(world, client, num_npc_vehicles, p)
    npc_walkers, npc_controllers = spawn_npc_pedestrians(world, client, num_npc_pedestrians, p)

    spawn_points = world.get_map().get_spawn_points()
    if spawn_idx == 0:
        spawn_pt = random.choice(spawn_points)
    else:
        spawn_pt = spawn_points[min(spawn_idx - 1, len(spawn_points) - 1)]

    vehicle = spawn_vehicle(world, client, EGO, spawn_pt, p)

    warmup_ticks = int(hz * 3)
    print(f"Warming up ({warmup_ticks} ticks / 3s)...")
    for i in range(warmup_ticks):
        world.tick()
        if i % int(hz) == 0:
            print(f"  Warmup: {i // int(hz) + 1}/3s")
    print("Warmup done — starting recording!")

    dvs_camera       = spawn_dvs_camera(world, vehicle, hz, p)
    grayscale_camera = spawn_grayscale_camera(world, vehicle, p)
    flow_camera      = spawn_flow_camera(world, vehicle, p)
    lidar_sensor     = spawn_lidar(world, vehicle, hz, p)

    spectator  = world.get_spectator()
    stop_event = threading.Event()
    threading.Thread(target=follow_camera,
                     args=(grayscale_camera, spectator, stop_event),
                     daemon=True).start()

    total_events = 0
    all_events, grayscale_frames, flow_frames = [], [], []
    lidar_estimator = LidarFlowEstimator(sim_hz=hz, flow_fps=p["gray_flow_fps"])

    dvs_anchor = {"t_ns": None, "sim_s": None}

    def save_dvs_events(image):
        nonlocal total_events
        dvs_events = np.frombuffer(image.raw_data, dtype=[
            ('x', np.uint16), ('y', np.uint16),
            ('t', np.int64),  ('pol', np.uint8)
        ])
        total_events += len(dvs_events)
        all_events.append(dvs_events.copy())
        if dvs_anchor["t_ns"] is None and len(dvs_events):
            dvs_anchor["t_ns"]  = int(dvs_events['t'].max())
            dvs_anchor["sim_s"] = float(image.timestamp)

    def save_grayscale(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        grayscale_frames.append(cv2.cvtColor(array, cv2.COLOR_RGB2GRAY).copy())

    def save_flow(image):
        flow_array = np.frombuffer(image.raw_data, dtype=np.float32)
        flow_frames.append(flow_array.reshape((image.height, image.width, 2)).copy())

    dvs_camera.listen(save_dvs_events)
    grayscale_camera.listen(save_grayscale)
    flow_camera.listen(save_flow)
    lidar_sensor.listen(lidar_estimator.callback)

    ticks       = 0
    total_ticks = int(hz * DUR)
    print(f"\n🎥 Recording {DUR}s @ {hz}Hz...")
    try:
        while ticks < total_ticks:
            world.tick()
            if ticks % 10 == 0:
                force_green_light(vehicle, world)
            ticks += 1
            if ticks % 500 == 0:
                pct = 100 * ticks / total_ticks
                print(f"  ⏱️  {ticks:>6}/{total_ticks} ({pct:5.1f}%) | {total_events:,} events")
    except KeyboardInterrupt:
        print("⏹️  Interrupted by user")

    print("\nSaving data...")

    if all_events:
        all_events_np = np.concatenate(all_events)
    else:
        all_events_np = np.array([], dtype=[
            ('x', np.uint16), ('y', np.uint16),
            ('t', np.int64),  ('pol', np.uint8)
        ])
    with h5py.File(os.path.join(output_dir, "events.h5"), 'w') as f:
        f.create_dataset('x',   data=all_events_np['x'],   compression='lzf')
        f.create_dataset('y',   data=all_events_np['y'],   compression='lzf')
        f.create_dataset('t',   data=all_events_np['t'],   compression='lzf')
        f.create_dataset('pol', data=all_events_np['pol'], compression='lzf')
        if dvs_anchor["t_ns"] is not None:
            f.attrs['anchor_event_t_ns'] = dvs_anchor["t_ns"]
            f.attrs['anchor_sim_time_s'] = dvs_anchor["sim_s"]
    print(f"  DVS: {total_events:,} events saved")

    for i, gray in enumerate(grayscale_frames):
        cv2.imwrite(os.path.join(grayscale_dir, f"gray_{i+1:04d}.png"), gray)
    print(f"  Grayscale: {len(grayscale_frames)} frames saved")

    if flow_frames:
        flow_all       = np.stack(flow_frames, axis=0)
        scaling_factor = hz / p["gray_flow_fps"]
        flow_all       = flow_all * scaling_factor
        N, H, W, _     = flow_all.shape
        x_flow_dist    = flow_all[..., 0]
        y_flow_dist    = -flow_all[..., 1]
        timestamps     = np.arange(N, dtype=np.float64) / p["gray_flow_fps"]
        with h5py.File(os.path.join(flow_dir, "flow.h5"), 'w') as f:
            f.create_dataset('x_flow_dist', data=x_flow_dist, compression='lzf')
            f.create_dataset('y_flow_dist', data=y_flow_dist, compression='lzf')
            f.create_dataset('timestamps',  data=timestamps,  compression='lzf')
        print(f"  Camera flow: {N} frames saved")

    if lidar_estimator.flow_frames:
        lf_all  = np.stack(lidar_estimator.flow_frames,  axis=0)
        lv_all  = np.stack(lidar_estimator.valid_frames, axis=0)
        N_lidar = lf_all.shape[0]
        t_start = np.asarray(lidar_estimator.frame_t0, dtype=np.float64)
        t_end   = np.asarray(lidar_estimator.frame_t1, dtype=np.float64)
        if dvs_anchor["t_ns"] is not None:
            t_first_event_sim = (dvs_anchor["sim_s"]
                                 - (dvs_anchor["t_ns"] - int(all_events_np['t'].min())) * 1e-9)
        else:
            t_first_event_sim = t_start[0]
        lts_all = t_start - t_first_event_sim
        with h5py.File(os.path.join(lidar_flow_dir, "flow.h5"), 'w') as f:
            f.create_dataset('x_flow_dist', data=lf_all[..., 0], compression='lzf')
            f.create_dataset('y_flow_dist', data=lf_all[..., 1], compression='lzf')
            f.create_dataset('valid_mask',  data=lv_all,          compression='lzf')
            f.create_dataset('timestamps',  data=lts_all,         compression='lzf')
            f.create_dataset('t_start_sim', data=t_start)
            f.create_dataset('t_end_sim',   data=t_end)
        print(f"  LiDAR flow:  {N_lidar} frames saved (sparse, with valid_mask)")

    total_pts   = lidar_estimator._total_points_seen
    removed_pts = lidar_estimator._dynamic_points_removed
    dynamic_pct = (100.0 * removed_pts / total_pts) if total_pts else 0.0
    print(f"  LiDAR dynamic-actor filter: {removed_pts:,}/{total_pts:,} points "
          f"removed ({dynamic_pct:.2f}%)")

    config_out = dict(p)
    config_out["lidar_dynamic_filter_stats"] = {
        "total_points_seen":      total_pts,
        "dynamic_points_removed": removed_pts,
        "dynamic_points_pct":     round(dynamic_pct, 4),
        "filtered_tags":          sorted(DYNAMIC_SEMANTIC_TAGS),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"  Config saved: config.json")
    print(f"  📁 Output: {os.path.abspath(output_dir)}")

    stop_event.set()

    all_actors = (
        [dvs_camera, grayscale_camera, flow_camera, lidar_sensor, vehicle]
        + npc_controllers
        + npc_walkers
        + npc_vehicles_list
    )
    cleanup(all_actors)
    cv2.destroyAllWindows()

    print(f"\n✅ Run {run_number}/{total_runs} complete | {ticks} ticks | {total_events:,} events")

def main():
    print("=" * 55)
    print("   CARLA DVS Batch Dataset Collector")
    print("=" * 55)

    if not os.path.exists(RUNS_CONFIG):
        print(f"❌ runs_config.json not found at: {RUNS_CONFIG}")
        print("Create it with a list of run configs. Example:")
        example = [
            {"map": "Town01", "weather": "ClearNoon",   "spawn_idx": 0, "npc_vehicles": 30, "npc_pedestrians": 20},
            {"map": "Town01", "weather": "ClearNight",  "spawn_idx": 0, "npc_vehicles": 30, "npc_pedestrians": 20},
            {"map": "Town03", "weather": "CloudyNoon",  "spawn_idx": 5, "npc_vehicles": 40, "npc_pedestrians": 25},
            {"map": "Town05", "weather": "ClearSunset", "spawn_idx": 0, "npc_vehicles": 20, "npc_pedestrians": 30},
        ]
        print(json.dumps(example, indent=2))
        sys.exit(1)

    with open(RUNS_CONFIG) as f:
        runs = json.load(f)

    print(f"📋 Loaded {len(runs)} runs from {RUNS_CONFIG}")
    print(f"⏱️  Estimated time: ~{len(runs) * (DURATION/60 + 3.5):.0f} minutes\n")

    for i, cfg in enumerate(runs):
        assert cfg["map"] in AVAILABLE_MAPS,    f"Run {i+1}: unknown map '{cfg['map']}'"
        assert cfg["weather"] in WEATHER_PRESETS, f"Run {i+1}: unknown weather '{cfg['weather']}'"
    print("✅ All configs valid\n")

    input("Press Enter to start all runs...\n")

    client = connect_carla()
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    failed_runs = []
    for i, cfg in enumerate(runs, 1):
        try:
            run_single(client, cfg, i, len(runs))
            print(f"Waiting 3 before next run...")
            time.sleep(3.0)
        except KeyboardInterrupt:
            print(f"\n⏹️  Batch interrupted at run {i}/{len(runs)}")
            break
        except Exception as e:
            print(f"\n❌ Run {i} failed: {e}")
            failed_runs.append((i, cfg, str(e)))
            print("Continuing to next run...")
            time.sleep(5.0)

    print("\n" + "=" * 55)
    print(f"🎉 Batch complete: {len(runs) - len(failed_runs)}/{len(runs)} runs succeeded")
    if failed_runs:
        print(f"❌ Failed runs: {[r[0] for r in failed_runs]}")
        for run_num, cfg, err in failed_runs:
            print(f"  Run {run_num} ({cfg['map']} / {cfg['weather']}): {err}")
    print(f"📁 Data saved to: {os.path.abspath(BASE_OUTPUT_DIR)}")
    print("=" * 55)

if __name__ == "__main__":
    main()
