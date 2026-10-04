"""Live CARLA walker-surrogate adapter for unicycle velocity commands.

Uses camera and semantic LiDAR returns, not omniscient pedestrian actor states.
The ego is a walker surrogate, not a validated TurtleBot dynamics model. The
server must advance asynchronously during policy inference. Exact branch reset
is not offered here: use the benchmark's deterministic scenario reconstruction.
"""
import os
import threading
import time
import numpy as np
import torch
from torch.nn import functional as F


class CarlaAdapter:
    def __init__(self, options):
        import carla
        self.carla = carla
        self.cfg = options["config"]
        self.client = carla.Client(os.environ.get("CARLA_HOST", "127.0.0.1"), int(os.environ.get("CARLA_PORT", "2000")))
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        if self.world.get_settings().synchronous_mode:
            raise RuntimeError("Live evaluation requires an asynchronously advancing CARLA server")
        actor_id = os.environ.get("TRACE_CARLA_EGO_ID")
        goal = os.environ.get("TRACE_CARLA_GOAL")
        if not actor_id or not goal:
            raise ValueError("Set TRACE_CARLA_EGO_ID and TRACE_CARLA_GOAL=x,y for each externally initialized episode")
        self.ego = self.world.get_actor(int(actor_id))
        if self.ego is None or not self.ego.type_id.startswith("walker.pedestrian"):
            raise ValueError("Ego must be an existing walker.pedestrian actor; stop its AI controller first")
        self.goal = np.array([float(x) for x in goal.split(",")])
        if self.goal.shape != (2,):
            raise ValueError("TRACE_CARLA_GOAL must contain x,y")
        self.lock = threading.Lock()
        self.rgb = self.cloud = None
        self.rgb_stamp = self.cloud_stamp = 0.0
        self.collision = False
        self.sensors = []
        self.previous = np.zeros(2, dtype=np.float32)
        self.previous_points = {}
        self.last_observed = -np.inf
        self.started = self.world.get_snapshot().timestamp.elapsed_seconds
        self.now = self.started
        self.yaw = np.deg2rad(self.ego.get_transform().rotation.yaw)
        try:
            blueprints = self.world.get_blueprint_library()
            camera = blueprints.find("sensor.camera.rgb")
            camera.set_attribute("image_size_x", "128"); camera.set_attribute("image_size_y", "128")
            camera.set_attribute("sensor_tick", str(self.cfg.dt))
            lidar = blueprints.find("sensor.lidar.ray_cast_semantic")
            lidar.set_attribute("range", "4.0"); lidar.set_attribute("rotation_frequency", str(1/self.cfg.dt))
            lidar.set_attribute("sensor_tick", str(self.cfg.dt))
            lidar.set_attribute("channels", "32"); lidar.set_attribute("points_per_second", "56000")
            for blueprint, callback in [(camera, self.on_rgb), (lidar, self.on_cloud),
                                         (blueprints.find("sensor.other.collision"), self.on_collision)]:
                sensor = self.world.spawn_actor(blueprint, carla.Transform(carla.Location(z=0.8)), attach_to=self.ego)
                self.sensors.append(sensor); sensor.listen(callback)
        except Exception:
            self.close()
            raise

    def on_rgb(self, image):
        pixels = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
        rgb = pixels[:, :, :3][:, :, ::-1].copy().transpose(2, 0, 1)/255.0
        with self.lock:
            self.rgb, self.rgb_stamp = rgb.astype(np.float32), float(image.timestamp)

    def on_cloud(self, cloud):
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("cos", "<f4"),
                          ("id", "<u4"), ("tag", "<u4")])
        points = np.frombuffer(cloud.raw_data, dtype=dtype).copy()
        with self.lock:
            self.cloud, self.cloud_stamp = points, float(cloud.timestamp)

    def on_collision(self, event):
        self.collision = True

    def timestamp(self):
        return self.now

    def observe(self):
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            snapshot = self.world.wait_for_tick(2.0)
            self.now = snapshot.timestamp.elapsed_seconds
            with self.lock:
                rgb, cloud, rs, ls = self.rgb, self.cloud, self.rgb_stamp, self.cloud_stamp
            if rgb is not None and cloud is not None and self.now-self.last_observed >= self.cfg.dt:
                if 0 <= self.now-min(rs, ls) <= 3*self.cfg.dt and min(rs, ls) > self.last_observed:
                    break
        else:
            raise TimeoutError("CARLA sensors are missing or stale")
        self.last_observed = min(rs, ls)
        transform = self.ego.get_transform()
        self.yaw = np.deg2rad(transform.rotation.yaw)
        # CARLA is left-handed: convert sensor y into right-handed robot-left y.
        xy = np.column_stack((cloud["x"], -cloud["y"]))
        useful = (cloud["z"] > -0.6) & (cloud["z"] < 1.5)
        bev = np.zeros((2, 64, 64), dtype=np.float32)
        obstacles = np.ones(16, dtype=np.float32)
        for point in xy[useful]:
            distance = np.linalg.norm(point)
            angle = np.arctan2(point[1], point[0])
            sector = int(((angle+np.pi) % (2*np.pi))/(2*np.pi)*16) % 16
            obstacles[sector] = min(obstacles[sector], distance/4)
            pixel = np.clip((point/4*31+32).astype(int), 0, 63)
            bev[0, pixel[1], pixel[0]] = 1
        people = np.zeros((8, 5), dtype=np.float32)
        detections = []
        for identity in np.unique(cloud["id"][cloud["tag"] == 4]):
            if int(identity) == self.ego.id:
                continue
            mask = (cloud["id"] == identity) & (cloud["tag"] == 4)
            center = xy[mask].mean(0)
            detections.append(center)
        # Position-only observations rely on history for motion; no actor velocities leak in.
        for i, point in enumerate(sorted(detections, key=np.linalg.norm)[:8]):
            people[i] = [point[0]/4, point[1]/4, 0, 0, 1]
            pixel = np.clip((point/4*31+32).astype(int), 0, 63)
            bev[1, pixel[1], pixel[0]] = 1
        delta = self.goal-np.array([transform.location.x, transform.location.y])
        forward = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        left = np.array([np.sin(self.yaw), -np.cos(self.yaw)])
        goal_local = np.array([delta@forward, delta@left])
        image = F.interpolate(torch.from_numpy(rgb)[None], size=(64, 64), mode="bilinear", align_corners=False)[0].numpy()
        return {"rgb": image, "bev": bev, "route": np.array([*goal_local/self.cfg.goal_scale,
                np.linalg.norm(delta)/self.cfg.goal_scale, np.arctan2(goal_local[1], goal_local[0])/np.pi], dtype=np.float32),
                "obstacles": obstacles, "pedestrians": people.flatten(),
                "robot": np.array([*self.previous, np.sin(-self.yaw), np.cos(-self.yaw)], dtype=np.float32),
                "previous_action": self.previous.copy()}

    def execute(self, command):
        # Positive angular velocity means left in the model, negative yaw in CARLA.
        v, w = np.asarray(command, dtype=float)
        heading = self.yaw-w*self.cfg.dt
        control = self.carla.WalkerControl()
        control.speed = float(v)
        control.direction = self.carla.Vector3D(float(np.cos(heading)), float(np.sin(heading)), 0.0)
        self.ego.apply_control(control)
        self.previous = np.array([v, w], dtype=np.float32)
        location = self.ego.get_location()
        return {"collision": bool(self.collision),
                "goal": bool(np.linalg.norm(self.goal-[location.x, location.y]) <= self.cfg.goal_tolerance),
                "timeout": self.now-self.started >= self.cfg.episode_seconds}

    def close(self):
        if getattr(self, "ego", None) is not None:
            self.ego.apply_control(self.carla.WalkerControl())
        for sensor in getattr(self, "sensors", []):
            try:
                sensor.stop(); sensor.destroy()
            except RuntimeError:
                pass
        self.sensors = []


def create(options):
    return CarlaAdapter(options)
