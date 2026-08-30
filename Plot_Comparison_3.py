import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# Make sure this matches your file name containing VehicleEnv
from Environment_2 import VehicleEnv

# Configuration
vers = ["Baseline", "seed=1", "seed=21", "seed=63", "seed=84", "seed=84_N=2.5e6", "gamma=0.95", "gamma=0.995"]

col1 = '#ff3a3a'
col2 = "#ff14ff"
col3 = "#2ca02c"
col4 = "#17e8ff"
col5 = "#f98612"
col6 = "#61d627"
col7 = "#0013bf"
col8 = "#8c564b"

line_colors = [col1, col2, col3, col4, col5, col6, col7, col8]  # Distinct colors per seed

# 1. Base environment initialization
base_env = make_vec_env(VehicleEnv, n_envs=1)
fig, ax = plt.subplots(figsize=(10, 8))

last_sim = None
last_env_target = None

# 2. Iterate through seeds and collect paths
for idx, ver in enumerate(vers):
    version = f"Brev_1_{ver}"
    print(f"Running evaluation episode for {version}...")

    # Load normalized environment and model for current seed
    env = VecNormalize.load(f"vec_normalize_{version}.pkl", base_env)
    env.training = False  
    env.norm_reward = False  
    model = SAC.load(f"vehicle_sac_model_{version}", env=env)

    x_hist = []
    y_hist = []

    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            break
        
        sim = env.get_attr("sim")[0]
        x_hist.append(sim.x)
        y_hist.append(sim.y)
        
    last_sim = sim
    last_env_target = env.get_attr("target")[0]

    # Plot line trajectory for this seed
    ax.plot(x_hist, y_hist, label=f"{ver}", color=line_colors[idx], linewidth=2.5, zorder=4)

# ====================================================
# 3. ADVANCED VISUALIZATION PLOT BACKGROUNDS
# ====================================================

# --- A. Terrain Heatmap Background ---
terrain = last_sim.terrain
grid_x, grid_y = np.meshgrid(np.linspace(-100, 1600, 200), np.linspace(-100, 1600, 200))
grid_z = terrain.height(grid_x, grid_y)

contour = ax.contourf(grid_x, grid_y, grid_z, cmap='terrain', alpha=0.6, levels=25)
cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, aspect=30, shrink=0.9)
cbar.set_label('Terrain Elevation (meters)', rotation=90, labelpad=-70)

# --- B. Plot Wind Direction ---
wind_x = 100 * np.cos(last_sim.wind_direction)
wind_y = 100 * np.array(np.sin(last_sim.wind_direction))
ax.quiver(1150, 1350, wind_x, wind_y, scale=800, color='black', width=0.007, label=f"Wind ({last_sim.wind_speed} m/s)")
ax.text(1025, 1125, "Wind Direction", color='black', fontsize=10, weight='bold')

# --- C. Plot Charging Stations with Proximity Zones ---
for i, station in enumerate(last_sim.charging_stations):
    ax.scatter(station[0], station[1], color='red', marker='P', s=120, zorder=5, 
               label="Charging Station" if i == 0 else "")
    circle = plt.Circle((station[0], station[1]), last_sim.station_radius, color='red', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)

# --- D. Standard Key Markers ---
ax.scatter([0], [0], color='green', marker='o', s=150, label="Start Position", zorder=6)
ax.scatter([last_env_target[0]], [last_env_target[1]], color='black', marker='X', s=150, label="Target Destination", zorder=6)

# --- E. Plot Cosmetics ---
ax.set_title("Agent Trajectories After Training", fontsize=14, weight='bold')
ax.set_xlabel("X Coordinate (meters)")
ax.set_ylabel("Y Coordinate (meters)")
ax.set_xlim(-50, 1400)
ax.set_ylim(-50, 1400)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
