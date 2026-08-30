
f1 = "logs/Brev_1_episodes.csv" # Baseline: seed = 42, gamma = 0.99
f2 = "logs/Brev_1_seed=1_episodes.csv"
f3 = "logs/Brev_1_seed=21_episodes.csv"
f4 = "logs/Brev_1_seed=63_episodes.csv"
f5 = "logs/Brev_1_seed=84_episodes.csv"
f6 = "logs/Brev_1_seed=84_ext_episodes.csv"
f7 = "logs/Brev_1_gamma=0.95_episodes.csv"
f8 = "logs/Brev_1_gamma=0.995_episodes.csv"

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(f1)
df2 = pd.read_csv(f2)
df3 = pd.read_csv(f3)
df4 = pd.read_csv(f4)
df5 = pd.read_csv(f5)
df6 = pd.read_csv(f6)
df7 = pd.read_csv(f7)
df8 = pd.read_csv(f8)

col1 = '#ff3a3a'
col2 = "#ff14ff"
col3 = "#2ca02c"
col4 = "#17e8ff"
col5 = "#f98612"
col6 = "#61d627"
col7 = "#0013bf"
col8 = "#8c564b"

success_rate_window = 200
energy_consumption_window = 200
reward_window = 200

df["success_rate"] = (df["success"].rolling(success_rate_window).mean())
df2["success_rate"] = (df2["success"].rolling(success_rate_window).mean())
df3["success_rate"] = (df3["success"].rolling(success_rate_window).mean())
df4["success_rate"] = (df4["success"].rolling(success_rate_window).mean())
df5["success_rate"] = (df5["success"].rolling(success_rate_window).mean())
df6["success_rate"] = (df6["success"].rolling(success_rate_window).mean())
df7["success_rate"] = (df7["success"].rolling(success_rate_window).mean())
df8["success_rate"] = (df8["success"].rolling(success_rate_window).mean())

df["energy_consumed"] = (df["energy_consumed"].rolling(energy_consumption_window).mean())
df2["energy_consumed"] = (df2["energy_consumed"].rolling(energy_consumption_window).mean())
df3["energy_consumed"] = (df3["energy_consumed"].rolling(energy_consumption_window).mean())
df4["energy_consumed"] = (df4["energy_consumed"].rolling(energy_consumption_window).mean())
df5["energy_consumed"] = (df5["energy_consumed"].rolling(energy_consumption_window).mean())
df6["energy_consumed"] = (df6["energy_consumed"].rolling(energy_consumption_window).mean())
df7["energy_consumed"] = (df7["energy_consumed"].rolling(energy_consumption_window).mean())
df8["energy_consumed"] = (df8["energy_consumed"].rolling(energy_consumption_window).mean())

df["episode_reward"] = (df["episode_reward"].rolling(reward_window).mean())
df2["episode_reward"] = (df2["episode_reward"].rolling(reward_window).mean())
df3["episode_reward"] = (df3["episode_reward"].rolling(reward_window).mean())
df4["episode_reward"] = (df4["episode_reward"].rolling(reward_window).mean())
df5["episode_reward"] = (df5["episode_reward"].rolling(reward_window).mean())
df6["episode_reward"] = (df6["episode_reward"].rolling(reward_window).mean())
df7["episode_reward"] = (df7["episode_reward"].rolling(reward_window).mean())
df8["episode_reward"] = (df8["episode_reward"].rolling(reward_window).mean())


# Success Rate Plot
plt.figure(figsize=(10, 5))

plt.plot(df["episode"], df["success_rate"], label="Baseline: seed=42, gamma=0.99, N=1.5e6", color=col1, linewidth=2)

plt.plot(df2["episode"], df2["success_rate"], label="seed=1", color=col2)
plt.plot(df3["episode"], df3["success_rate"], label="seed=21", color=col3)
plt.plot(df4["episode"], df4["success_rate"], label="seed=63", color=col4)
plt.plot(df5["episode"], df5["success_rate"], label="seed=84", color=col5)
plt.plot(df6["episode"], df6["success_rate"], label="seed=84, N=2.5e6", color=col6, linestyle='--')

plt.plot(df7["episode"], df7["success_rate"], label="gamma=0.95", color=col7, linestyle=':')
plt.plot(df8["episode"], df8["success_rate"], label="gamma=0.995", color=col8, linestyle=':')
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Training Success Rate")
plt.legend()
plt.grid(True)

plt.show()



# # Energy Consumption Plot
# plt.figure(figsize=(10, 5))

# plt.plot(df["episode"], df["energy_consumed"], label="Baseline: seed=42, gamma=0.99", color=col1)
# plt.plot(df2["episode"], df2["energy_consumed"], label="seed=1, gamma=0.99, N=1.5e6", color=col2)
# plt.plot(df3["episode"], df3["energy_consumed"], label="seed=21, gamma=0.99, N=1.5e6", color=col3)
# plt.plot(df4["episode"], df4["energy_consumed"], label="seed=63, gamma=0.99, N=1.5e6", color=col4)
# plt.plot(df5["episode"], df5["energy_consumed"], label="seed=84, gamma=0.99, N=1.5e6", color=col5)
# plt.plot(df6["episode"], df6["energy_consumed"], label="seed=84, gamma=0.99, N=2.5e6", color=col6)
# plt.plot(df7["episode"], df7["energy_consumed"], label="seed=42, gamma=0.95, N=1.5e6", color=col7)
# plt.plot(df8["episode"], df8["energy_consumed"], label="seed=42, gamma=0.995, N=1.5e6", color=col8)
# plt.xlabel("Episode")
# plt.ylabel("Energy Consumed (battery fractions)")
# plt.title("Energy Consumption During Training")
# plt.legend()
# plt.grid(True)

# plt.show()


# Episode Reward Plot
plt.figure(figsize=(10, 5))

plt.plot(df["episode"], df["episode_reward"], label="Baseline: seed=42, gamma=0.99, N=1.5e6", color=col1, linewidth=2)

plt.plot(df2["episode"], df2["episode_reward"], label="seed=1", color=col2)
plt.plot(df3["episode"], df3["episode_reward"], label="seed=21", color=col3)
plt.plot(df4["episode"], df4["episode_reward"], label="seed=63", color=col4)
plt.plot(df5["episode"], df5["episode_reward"], label="seed=84", color=col5)
plt.plot(df6["episode"], df6["episode_reward"], label="seed=84, N=2.5e6", color=col6, linestyle='--')

plt.plot(df7["episode"], df7["episode_reward"], label="gamma=0.95", color=col7, linestyle=':')
plt.plot(df8["episode"], df8["episode_reward"], label="gamma=0.995", color=col8, linestyle=':')

plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Episode Reward During Training")
plt.legend()
plt.grid(True)

plt.show()
