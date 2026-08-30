import os
import csv
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogger(BaseCallback):

    def __init__(self, csv_path="training_results.csv", verbose=0):
        super().__init__(verbose)

        self.csv_path = csv_path
        self.episode_count = 0

        # Create CSV file and header
        os.makedirs(
            os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".",
            exist_ok=True
        )

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "episode",
                "timesteps",
                "episode_reward",
                "episode_duration",
                "energy_consumed",
                "final_distance",
                "final_speed",
                "remaining_battery",
                "success"
            ])

    def _on_step(self):

        # With VecEnv, infos is a list
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, done in enumerate(dones):

            if done:

                info = infos[i]

                self.episode_count += 1

                episode_reward = self.locals["rewards"][i]

                # Try to obtain SB3's episode statistics
                if "episode" in info:

                    episode_reward = info["episode"]["r"]

                    episode_duration = info["episode"]["l"]

                else:

                    episode_duration = info.get("time", np.nan)

                energy_consumed = info.get(
                    "net_energy_used",
                    np.nan
                )

                final_distance = info.get(
                    "distance",
                    np.nan
                )

                final_speed = info.get(
                    "speed",
                    np.nan
                )

                remaining_battery = info.get(
                    "battery",
                    np.nan
                )

                success = info.get(
                    "success",
                    0
                )

                with open(self.csv_path, "a", newline="") as f:

                    writer = csv.writer(f)

                    writer.writerow([
                        self.episode_count,
                        self.num_timesteps,
                        episode_reward,
                        episode_duration,
                        energy_consumed,
                        final_distance,
                        final_speed,
                        remaining_battery,
                        success
                    ])

        return True