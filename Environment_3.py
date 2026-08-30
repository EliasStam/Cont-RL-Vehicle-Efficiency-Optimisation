import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Simulator_3 import VehicleSimulator


class VehicleEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.sim = VehicleSimulator()
        self.target = np.array([800.0, 1300.0], dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        low_obs = np.array(
            [0, -np.pi, 0, 0, 0, 0, -np.pi, 0, 0, -np.pi, 0, 0, -np.pi, 0, 0, -np.pi, 0],
            dtype=np.float32,
        )
        high_obs = np.array(
            [1, np.pi, 1, 1, 1, 1, np.pi, 1, 1, np.pi, 1, 1, np.pi, 1, 1, np.pi, 1],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )

    def _get_obs(self):
        # Target information
        dx = self.target[0] - self.sim.x
        dy = self.target[1] - self.sim.y

        dist_to_target = np.sqrt(dx**2 + dy**2) / 3000
        target_heading = np.arctan2(dy, dx)
        target_bearing = target_heading - self.sim.heading
        target_bearing = np.arctan2(
            np.sin(target_bearing), np.cos(target_bearing)
        )

        # Basic vehicle state
        speed = self.sim.v / 20
        battery_fraction = self.sim.battery_energy / self.sim.max_battery
        time_fraction = self.sim.time / 200

        # Charging station information
        station_observations = []
        for i, station in enumerate(self.sim.charging_stations):
            sdx = station[0] - self.sim.x
            sdy = station[1] - self.sim.y

            station_dist = np.sqrt(sdx**2 + sdy**2) / 3000
            station_heading = np.arctan2(sdy, sdx)
            station_bearing = station_heading - self.sim.heading
            station_bearing = np.arctan2(
                np.sin(station_bearing), np.cos(station_bearing)
            )

            station_visited = float(self.sim.stations_visited[i])
            station_observations.extend(
                [station_dist, station_bearing, station_visited]
            )

        return np.array(
            [
                dist_to_target,
                target_bearing,
                speed,
                battery_fraction,
                time_fraction,
                *station_observations[:12],
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        obs = self._get_obs()
        self.episode_reward = 0.0
        self.episode_success = False
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        throttle, steering = action

        # State before physics step
        prev_x, prev_y = self.sim.x, self.sim.y
        prev_battery_fraction = (
            self.sim.battery_energy / self.sim.max_battery
        )
        prev_dist = np.sqrt(
            (self.target[0] - prev_x) ** 2 + (self.target[1] - prev_y) ** 2
        )
        prev_stations_visited = self.sim.stations_visited.copy()

        # Physics step
        self.sim.step(throttle, steering)
        self.sim.heading = np.arctan2(
            np.sin(self.sim.heading), np.cos(self.sim.heading)
        )

        obs = self._get_obs()

        dist = obs[0] * 3000
        speed = obs[2] * 20
        battery_fraction = obs[3]
        time = obs[4] * 200.0

        # Progress & Energy
        progress = prev_dist - dist
        if not np.isfinite(progress):
            progress = 0.0

        step_energy_net = prev_battery_fraction - battery_fraction
        cumulative_energy_used = self.sim.energy_used / self.sim.max_battery

        terminated = False
        truncated = False
        reward = 0.0

        # Rewards
        reward += 2.0 * progress

        if step_energy_net > -0.05:
            reward -= 100 * step_energy_net

        if dist < 50.0:
            reward -= 0.025 * speed

        # Charging Bonus
        for i in range(4):
            newly_charged = (
                not prev_stations_visited[i] and self.sim.stations_visited[i]
            )
            if newly_charged:
                reward += 100
                self.sim.has_charged = True
                print(
                    f"Battery CHARGED! ({self.sim.x:.2f}, {self.sim.y:.2f}), "
                    f"battery: {self.sim.battery_energy:.2f} J, time: {self.sim.time:.2f} s"
                )

        self.episode_reward += reward

        # Termination & Success
        if dist < 5.0:
            print("Goal overshoot!!!")
            reward += 0.2
            self.episode_reward += 1
            if speed < 1.0:
                success_reward = 2 * (100 - cumulative_energy_used * 10)
                reward += success_reward
                print(
                    f"Success achieved: ({self.sim.x:.2f}, {self.sim.y:.2f}), "
                    f"speed: {speed:.2f} m/s, ep_reward: {(self.episode_reward + success_reward):.2f}, "
                    f"battery: {battery_fraction:.2f} %, time: {self.sim.time:.2f} s, "
                    f"energy used: {cumulative_energy_used:.2f} batteries"
                )
                self.sim.has_charged = False
                self.episode_success = True
                terminated = True

        elif self.sim.battery_energy <= 0:
            bdp = 400
            reward -= bdp
            status = "post-charge" if self.sim.has_charged else "pre-charge"
            print(
                f"Battery {status}: ({self.sim.x:.2f}, {self.sim.y:.2f}), "
                f"speed: {speed:.2f} m/s, ep_reward: {(self.episode_reward - bdp):.2f}, "
                f"time: {self.sim.time:.2f} s, energy used: {cumulative_energy_used:.2f} batteries"
            )
            self.sim.has_charged = False
            terminated = True

        elif time >= 200:
            status = "post-charge" if self.sim.has_charged else "pre-charge"
            print(
                f"Timeout {status}: ({self.sim.x:.2f}, {self.sim.y:.2f}), "
                f"speed: {speed:.2f} m/s, ep_reward: {self.episode_reward:.2f}, "
                f"battery: {self.sim.battery_energy:.2f} J, energy used: {cumulative_energy_used:.2f} batteries"
            )
            self.sim.has_charged = False
            truncated = True

        info = {
            "distance": dist,
            "battery": self.sim.battery_energy,
            "speed": speed,
            "time": time,
            "net_energy_used": cumulative_energy_used,
            "success": float(self.episode_success),
        }

        return obs, reward, terminated, truncated, info