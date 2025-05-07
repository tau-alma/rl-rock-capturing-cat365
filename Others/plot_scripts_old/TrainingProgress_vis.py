import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# Load all monitor logs
log_files = glob.glob("results/excavator365-RockCapturing/*/2025-04-04_11-28-57/monitor_logs/*.csv", recursive=True)

all_data = []
for file in log_files:
    df = pd.read_csv(file, skiprows=1)  # Skip first row (header)
    df = df[['l', 'r']]  # Keep only episode length (ep_len) and reward (ep_rew)
    all_data.append(df)

# Ensure all data has the same number of episodes
min_episodes = min(len(df) for df in all_data)
trimmed_rewards = np.array([df['r'][:min_episodes] for df in all_data])

# Compute mean and standard deviation per episode
mean_rewards = np.mean(trimmed_rewards, axis=0)
std_rewards = np.std(trimmed_rewards, axis=0)
episode_numbers = np.arange(1, min_episodes + 1)  # Episode numbers

# Plot with margin
plt.figure(figsize=(10, 5))
plt.plot(episode_numbers, mean_rewards, label="Mean Reward (12 envs)", color="blue")
plt.fill_between(episode_numbers, mean_rewards - std_rewards, mean_rewards + std_rewards, color="blue", alpha=0.2, label="±1 Std Dev")

plt.xlabel("Episodes")
plt.ylabel("Episode Reward")
plt.title("Training Progress with Variability (Mean ± Std) - By Episode")
plt.legend()
plt.grid()
plt.show()
