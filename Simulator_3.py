import numpy as np


class Terrain:
    """Simple analytic terrain. z = h(x,y)"""

    def height(self, x, y):
        return 20 * np.sin(0.01 * x) + 10 * np.cos(0.015 * y)

    def gradient(self, x, y):
        """Returns: dz_dx, dz_dy"""
        dz_dx = 20 * 0.01 * np.cos(0.01 * x)
        dz_dy = -10 * 0.015 * np.sin(0.015 * y)
        return dz_dx, dz_dy


class VehicleSimulator:

    def __init__(self):
        # Vehicle parameters
        self.mass = 1500.0  # kg
        self.g = 9.81
        self.Cd = 0.28
        self.area = 2.2
        self.rho_air = 1.225
        self.Crr = 0.01
        self.max_drive_force = 4000.0
        self.max_regen_force = 3000.0
        self.regen_efficiency = 0.7
        self.max_steer_rate = np.deg2rad(30)
        self.dt = 0.1
        self.terrain = Terrain()
        self.station_radius = 30
        self.max_battery = 0.8e6

        # Constant wind
        self.wind_speed = 10.0
        self.wind_direction = np.deg2rad(-90)

        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.heading = 0.0
        self.battery_energy = self.max_battery
        self.time = 0.0
        self.energy_used = 0.0
        self.has_charged = False

        self.charging_stations = [
            np.array([250.0, 600.0]),
            np.array([600.0, 700.0]),
            np.array([800.0, 400.0]),
            np.array([1000.0, 800.0]),
        ]
        self.stations_visited = [False] * len(self.charging_stations)

        return self.get_state()

    def get_state(self):
        return np.array(
            [self.x, self.y, self.v, self.heading, self.battery_energy],
            dtype=np.float32,
        )

    def step(self, throttle, steering):
        throttle = np.clip(throttle, -1.0, 1.0)
        steering = np.clip(steering, -1.0, 1.0)
        dt = self.dt

        # Terrain slope
        dz_dx, dz_dy = self.terrain.gradient(self.x, self.y)
        travel_direction = np.array(
            [np.cos(self.heading), np.sin(self.heading)]
        )
        slope_along_path = (
            dz_dx * travel_direction[0] + dz_dy * travel_direction[1]
        )
        gravity_force = (
            self.mass
            * self.g
            * slope_along_path
            / np.sqrt(1 + slope_along_path**2)
        )

        # Drive / regen
        if throttle >= 0:
            drive_force = throttle * self.max_drive_force
            regen_force = 0.0
        else:
            drive_force = 0.0
            regen_force = abs(throttle) * self.max_regen_force

        # Wind
        wind_vector = np.array(
            [
                self.wind_speed * np.cos(self.wind_direction),
                self.wind_speed * np.sin(self.wind_direction),
            ]
        )
        wind_along_vehicle = np.dot(wind_vector, travel_direction)
        relative_speed = self.v - wind_along_vehicle

        # Resistive forces
        drag_force = (
            0.5
            * self.rho_air
            * self.Cd
            * self.area
            * relative_speed
            * abs(relative_speed)
        )
        rolling_force = self.Crr * self.mass * self.g

        # Longitudinal dynamics
        net_force = (
            drive_force
            - regen_force
            - drag_force
            - rolling_force
            - gravity_force
        )
        acceleration = net_force / self.mass
        self.v += acceleration * dt

        MAX_SPEED = 20.0  # m/s
        self.v = np.clip(self.v, 0.0, MAX_SPEED)

        # Heading and position dynamics
        self.heading += steering * self.max_steer_rate * dt
        self.x += self.v * np.cos(self.heading) * dt
        self.y += self.v * np.sin(self.heading) * dt

        # Battery model
        if drive_force > 0:
            energy_delta = drive_force * self.v * dt
            self.battery_energy -= energy_delta
            self.energy_used += energy_delta

        if regen_force > 0:
            energy_delta = regen_force * self.v * self.regen_efficiency * dt
            self.battery_energy += energy_delta
            self.energy_used -= energy_delta

        # Charging station logic
        for i, station in enumerate(self.charging_stations):
            dist_to_station = np.linalg.norm(
                np.array([self.x, self.y]) - station
            )
            if (
                dist_to_station < self.station_radius
                and not self.stations_visited[i]
            ):
                self.battery_energy = self.max_battery
                self.stations_visited[i] = True
                self.has_charged = True
                break

        self.battery_energy = max(0.0, self.battery_energy)
        self.time += dt

        return self.get_state()