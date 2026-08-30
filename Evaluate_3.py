import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Make sure this matches your file name containing VehicleEnv
from Environment_3 import VehicleEnv

version = "Brev_2"

# 1. Recreate and wrap the environment
env = make_vec_env(VehicleEnv, n_envs=1)
env = VecNormalize.load(f"vec_normalize_{version}.pkl", env)
env.training = False
env.norm_reward = False

# 2. Load the trained agent
model = SAC.load(f"vehicle_sac_model_{version}", env=env)

x_hist = []
y_hist = []
v_hist = []
battery_hist = []

obs = env.reset()
done = False

print("Running evaluation episode...")
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if done[0]:
        break

    sim = env.get_attr("sim")[0]
    x_hist.append(sim.x)
    y_hist.append(sim.y)
    v_hist.append(sim.v)
    battery_hist.append(sim.battery_energy)

print("\n--- Episode Summary ---")
print(f"Final Distance to Target: {info[0]['distance']:.2f} m")
print(f"Final Speed: {info[0]['speed']:.2f} m/s")
print(f"Remaining Battery: {info[0]['battery']/1e6:.2f} MJ")
print(f"Time taken: {info[0]['time']:.2f} s")

# 3. Visualization Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Terrain Heatmap Background
terrain = sim.terrain
grid_x, grid_y = np.meshgrid(
    np.linspace(-100, 1600, 200), np.linspace(-100, 1600, 200)
)
grid_z = terrain.height(grid_x, grid_y)

contour = ax.contourf(
    grid_x, grid_y, grid_z, cmap="terrain", alpha=0.6, levels=25
)
cbar = fig.colorbar(
    contour, ax=ax, orientation="vertical", pad=-0.05, aspect=30, shrink=0.8
)
cbar.set_label("Terrain Elevation (meters)", rotation=90, labelpad=-49)

# Wind Direction
wind_x = 100 * np.cos(sim.wind_direction)
wind_y = 100 * np.sin(sim.wind_direction)
ax.quiver(
    1150,
    1350,
    wind_x,
    wind_y,
    scale=800,
    color="black",
    width=0.007,
    label=f"Wind ({sim.wind_speed} m/s)",
)
ax.text(
    1025, 1125, "Wind Direction", color="black", fontsize=10, weight="bold"
)

# Charging Stations
for i, station in enumerate(sim.charging_stations):
    ax.scatter(
        station[0],
        station[1],
        color="red",
        marker="P",
        s=120,
        zorder=5,
        label="Charging Station" if i == 0 else "",
    )
    circle = plt.Circle(
        (station[0], station[1]),
        sim.station_radius,
        color="red",
        fill=False,
        linestyle="--",
        alpha=0.5,
    )
    ax.add_patch(circle)

# Trajectory Path Color-coded by Battery Level
battery_frac = np.array(battery_hist) / sim.max_battery
points = np.array([x_hist, y_hist]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(
    segments,
    cmap="plasma",
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3.5,
    zorder=3,
)
lc.set_array(battery_frac)
line = ax.add_collection(lc)
cbar2 = fig.colorbar(
    line, ax=ax, orientation="vertical", pad=0.05, aspect=30, shrink=0.8
)
cbar2.set_label("Battery Charge Fraction", rotation=90, labelpad=-34)

# Key Markers & Formatting
ax.scatter(
    [0], [0], color="green", marker="o", s=150, label="Start Position", zorder=6
)
env_target = env.get_attr("target")[0]
ax.scatter(
    [env_target[0]],
    [env_target[1]],
    color="black",
    marker="X",
    s=150,
    label="Target Destination",
    zorder=6,
)

ax.set_title(
    f"Agent Trajectory Profile (Model: {version})", fontsize=14, weight="bold"
)
ax.set_xlabel("X Coordinate (meters)")
ax.set_ylabel("Y Coordinate (meters)")
ax.set_xlim(-50, 1400)
ax.set_ylim(-50, 1400)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="lower right")

plt.tight_layout()
plt.show()