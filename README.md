# Cont-RL-Vehicle-Efficiency-Optimisation

# Autonomous EV Path Planning & Control via Reinforcement Learning

This repository implements a custom Gymnasium environment and a 2D kinematic vehicle simulator to train autonomous electric vehicle agents using Soft Actor-Critic (SAC) via Stable-Baselines3. The agent must navigate 2D terrain, manage energy consumption, account for atmospheric wind forces, and utilize charging stations to reach a target destination.

## Key Features

* **Kinematic Vehicle Physics**: Custom simulator accounting for mass, aerodynamic drag, rolling resistance, wind vectors, elevation gradient/slope forces, and regenerative braking.


* **Energy & Charging Management**: Tracks battery depletion and dynamic charging station interactions.


* **Gymnasium Environment**: Continuous action space (`throttle`, `steering`) and rich observation space.


* **Training & Logging Pipelines**: Integrated training scripts using SAC with custom callbacks to log episode metrics to CSV files.


* **Visualization Tools**: Evaluation scripts to plot agent trajectories against 2D terrain contour maps, wind vectors, and charging zones.



## Repository Structure

```text
.
├── Simulator_3.py          # Kinematic vehicle model and analytic terrain implementation
├── Environment_3.py        # Gymnasium environment wrapper with custom reward functions
├── TrainingLogger_3.py     # Stable-Baselines3 callback for logging training stats
├── Train_3.py              # Model training script using SAC
├── Plot_Logs_3.py          # Trajectory plotting script for multi-model evaluation
└── Plot_Comparison_3.py    # Training metrics visualization (Success Rate, Rewards, etc.)

```

## Setup & Installation

Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install numpy matplotlib gymnasium stable-baselines3 pandas

```

## Quick Start

### 1. Training an Agent

To start training a Soft Actor-Critic (SAC) model:

```bash
python Train_3.py

```

This runs training for `1,500,000` timesteps, normalizes environment observations, saves log metrics to `logs/`, and outputs the trained model `.zip` file alongside its `VecNormalize` statistics.

### 2. Plotting Training Metrics

To generate training curve comparisons across different seeds or hyperparameters:

```bash
python Plot_Logs_3.py

```

### 3. Visualizing Trajectories

To plot trained vehicle trajectories on top of terrain contours:

```bash
python Plot_Comparison_3.py

```
