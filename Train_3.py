from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from Environment_3 import VehicleEnv
from TrainingLogger_3 import EpisodeLogger


version = "Brev_2"

env = make_vec_env(VehicleEnv, n_envs=1)
env = VecNormalize(env, norm_obs=True, norm_reward=False) 
logger_callback = EpisodeLogger(csv_path=f"logs/{version}_episodes.csv")

model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    seed=42,
    learning_rate=3e-4,
    buffer_size=1_500_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    learning_starts=10_000,
    tensorboard_log="./logs/"
)

model.learn(
    total_timesteps=1_500_000,
    progress_bar=True,
    tb_log_name=f"SAC_{version}", 
    callback=logger_callback
)

model.save(f"vehicle_sac_model_{version}")
env.save(f"vec_normalize_{version}.pkl")