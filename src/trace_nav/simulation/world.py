"""Small deterministic-restore 2-D crowd simulator for algorithm development.

The raster observations are synthetic sensor proxies, not CARLA camera images.
This simulator is not C2N-Bench and must not be reported as that benchmark.
"""
import copy
import heapq
import math
import numpy as np


class World:
    def __init__(self, cfg, seed=0, pedestrians=8):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.size = 12.0
        self.resolution = self.size/64
        self.robot = np.array([1.0, 1.0, 0.0])
        self.goal = np.array([11.0, 11.0])
        self.action = np.zeros(2)
        self.time = 0.0
        self.collision = False
        self.obstacles = [(4.0, 5.0, 0.6), (7.0, 7.0, 0.7), (8.0, 3.0, 0.6)]
        positions = []
        for _ in range(pedestrians):
            for attempt in range(1000):
                xy = self.rng.uniform(2, 10, 2)
                if not self.blocked(*xy, radius=0.25) and all(np.linalg.norm(xy-p) > 0.5 for p in positions):
                    positions.append(xy)
                    break
            else:
                raise ValueError("Could not place pedestrians without initial overlaps")
        self.people = np.column_stack((np.asarray(positions).reshape(-1, 2),
                                      self.rng.uniform(-0.5, 0.5, (pedestrians, 2))))
        self._distance_field()

    def blocked(self, x, y, radius=0):
        if min(x, y) < radius or max(x, y) > self.size-radius:
            return True
        return any(math.hypot(x-a, y-b) <= r+radius for a, b, r in self.obstacles)

    def _distance_field(self):
        n = 64
        free = np.array([[not self.blocked((x+0.5)*self.resolution, (y+0.5)*self.resolution,
                                          self.cfg.robot_radius) for x in range(n)] for y in range(n)])
        gx, gy = np.clip((self.goal/self.resolution).astype(int), 0, 63)
        self.distances = np.full((n, n), np.inf)
        self.distances[gy, gx] = 0
        heap = [(0.0, int(gy), int(gx))]
        while heap:
            distance, y, x = heapq.heappop(heap)
            if distance != self.distances[y, x]:
                continue
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                xx, yy = x+dx, y+dy
                if 0 <= xx < n and 0 <= yy < n and free[yy, xx]:
                    new = distance+self.resolution
                    if new < self.distances[yy, xx]:
                        self.distances[yy, xx] = new
                        heapq.heappush(heap, (new, yy, xx))

    def path_distance(self):
        x, y = np.clip((self.robot[:2]/self.resolution).astype(int), 0, 63)
        return float(min(self.distances[y, x], self.cfg.distance_cap))

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, state):
        self.__dict__ = copy.deepcopy(state)

    def intervene(self, kind):
        if kind == "reverse_pedestrian" and len(self.people):
            index = np.argmin(np.linalg.norm(self.people[:, :2]-self.robot[:2], axis=1))
            self.people[index, 2:] *= -1
        elif kind == "block_passage":
            self.obstacles.append((6.0, 6.0, 0.6))
            self._distance_field()
        elif kind != "reverse_pedestrian":
            raise ValueError(f"Unknown intervention: {kind}")

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def valid_initial_state(self):
        if self.blocked(*self.robot[:2], self.cfg.robot_radius):
            return False
        if any(self.blocked(*p[:2], radius=0.25) for p in self.people):
            return False
        distances = np.linalg.norm(self.people[:, :2]-self.robot[:2], axis=1)
        return not bool(np.any(distances <= 0.25+self.cfg.robot_radius))

    def step(self, command):
        cfg = self.cfg
        previous = self.action.copy()
        desired = np.clip(command, [0, -cfg.max_w], [cfg.max_v, cfg.max_w])
        change = np.array([cfg.max_a, cfg.max_alpha])*cfg.dt
        self.action = previous+np.clip(desired-previous, -change, change)
        # Substeps reduce tunneling through small objects and collision surfaces.
        hit = self.collision
        for _ in range(5):
            dt = cfg.dt/5
            self.robot[2] += self.action[1]*dt
            self.robot[:2] += self.action[0]*dt*np.array([np.cos(self.robot[2]), np.sin(self.robot[2])])
            old_positions = self.people[:, :2].copy()
            self.people[:, :2] += self.people[:, 2:]*dt
            for i, person in enumerate(self.people):
                if self.blocked(*person[:2], radius=0.25):
                    self.people[i, :2] = old_positions[i]
                    self.people[i, 2:] *= -1
            for axis in (0, 1):
                mask = (self.people[:, axis] < 0.4) | (self.people[:, axis] > self.size-0.4)
                self.people[mask, axis+2] *= -1
                self.people[:, axis] = np.clip(self.people[:, axis], 0.4, self.size-0.4)
            distances = np.linalg.norm(self.people[:, :2]-self.robot[:2], axis=1)-0.25-cfg.robot_radius
            hit |= self.blocked(*self.robot[:2], cfg.robot_radius) or bool(np.any(distances <= 0))
            if hit:
                break
        self.collision = hit
        self.time += cfg.dt
        return {"collision": bool(hit), "goal": bool(np.linalg.norm(self.goal-self.robot[:2]) <= cfg.goal_tolerance),
                "timeout": self.time >= cfg.episode_seconds}

    def local(self, points):
        theta = self.robot[2]
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return (np.asarray(points)-self.robot[:2]) @ rotation

    def visible(self, xy):
        delta = xy-self.robot[:2]
        if np.linalg.norm(delta) > 4.0:
            return False
        for fraction in np.linspace(0.05, 0.95, 24):
            if self.blocked(*(self.robot[:2]+fraction*delta)):
                return False
        return True

    def observe(self):
        theta = self.robot[2]
        lidar = []
        bev = np.zeros((2, 64, 64), dtype=np.float32)
        for angle in np.linspace(-np.pi, np.pi, 16, endpoint=False):
            direction = np.array([np.cos(theta+angle), np.sin(theta+angle)])
            measured = 4.0
            for distance in np.linspace(0.05, 4.0, 80):
                xy = self.robot[:2]+distance*direction
                if self.blocked(*xy):
                    measured = distance
                    break
            lidar.append(measured/4.0)
            pixel = np.clip((np.array([np.cos(angle), np.sin(angle)])*measured/4*31+32).astype(int), 0, 63)
            if measured < 4:
                bev[0, pixel[1], pixel[0]] = 1
        people = sorted((p for p in self.people if self.visible(p[:2])),
                        key=lambda p: np.linalg.norm(p[:2]-self.robot[:2]))[:8]
        features = np.zeros((8, 5), dtype=np.float32)
        for i, p in enumerate(people):
            xy = self.local(p[:2]); velocity = self.local(p[:2]+p[2:])-xy
            features[i] = [xy[0]/4, xy[1]/4, velocity[0], velocity[1], 1]
            pixel = np.clip((xy/4*31+32).astype(int), 0, 63)
            bev[1, pixel[1], pixel[0]] = 1
        goal = self.local(self.goal)
        rgb = np.stack((bev[0], bev[1], np.zeros((64, 64), dtype=np.float32)))
        return {"rgb": rgb, "bev": bev, "route": np.array([*goal/self.cfg.goal_scale,
                 np.linalg.norm(goal)/self.cfg.goal_scale, math.atan2(goal[1], goal[0])/np.pi], dtype=np.float32),
                "obstacles": np.asarray(lidar, dtype=np.float32), "pedestrians": features.flatten(),
                "robot": np.array([*self.action, np.sin(theta), np.cos(theta)], dtype=np.float32),
                "previous_action": self.action.astype(np.float32)}

    def expert_sequence(self):
        """Geodesic local teacher for initialization, not an oracle benchmark baseline."""
        saved = self.snapshot()
        result = []
        for _ in range(self.cfg.horizon):
            x, y = np.clip((self.robot[:2]/self.resolution).astype(int), 1, 62)
            neighbors = [(self.distances[y+dy, x+dx], x+dx, y+dy)
                         for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
            _, xx, yy = min(neighbors)
            target = np.array([(xx+0.5)*self.resolution, (yy+0.5)*self.resolution])
            heading = math.atan2(*(target-self.robot[:2])[::-1])
            error = (heading-self.robot[2]+np.pi) % (2*np.pi)-np.pi
            command = [self.cfg.max_v*max(0, np.cos(error)), np.clip(2*error, -self.cfg.max_w, self.cfg.max_w)]
            self.step(command)
            result.append(self.action.copy())
        self.restore(saved)
        return np.asarray(result, dtype=np.float32)


def outcome(world, commands):
    """Measure normalized [collision, path distance, proximity, motion] targets."""
    cfg = world.cfg
    proximity, motion, collision = 0.0, 0.0, False
    for command in commands:
        if collision:
            proximity += 1.0
            continue
        previous = world.action.copy()
        terminal = world.step(command)
        difference = (world.action-previous)/(np.array([cfg.max_a, cfg.max_alpha])*cfg.dt)
        motion += 0.5*float(np.square(difference).sum())
        collision = terminal["collision"]
        if collision:
            proximity += 1.0
        elif len(world.people):
            distances = np.linalg.norm(world.people[:, :2]-world.robot[:2], axis=1)-0.25-cfg.robot_radius
            proximity += float(np.clip(1-distances/cfg.proximity_radius, 0, 1).max())
    return np.array([float(collision), world.path_distance()/cfg.goal_scale,
                     proximity/len(commands), motion/len(commands)], dtype=np.float32)
