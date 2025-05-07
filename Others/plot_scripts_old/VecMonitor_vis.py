import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the monitor log file
# monitor_log_file = os.path.join(vec_monitor_log_dir, 'monitor.csv')

# Load the CSV into a pandas DataFrame
df = pd.read_csv("results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/2025-04-04_11-28-57/vec_monitor_logs/monitor.csv", 
                 comment="#")

print(df.columns)


# Inspect the first few rows
print(df.head())


# Plot the episode reward over time
plt.figure(figsize=(10, 5))
plt.plot(df['t'], df['r'], label='Episode Reward')
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('Episode Reward Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the episode length over time
plt.figure(figsize=(10, 5))
plt.plot(df['t'], df['l'], label='Episode Length', color='orange')
plt.xlabel('Timesteps')
plt.ylabel('Episode Length')
plt.title('Episode Length Over Time')
plt.legend()
plt.grid(True)
plt.show()
