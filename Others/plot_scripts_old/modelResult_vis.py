import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
csv_path = 'results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/2025-04-04_11-28-57/model_log/progress.csv'
df = pd.read_csv(csv_path)

# Check for NaN values
print(df.isna().sum())

# Optionally drop or fill NaN values
df = df.dropna()  # or df = df.fillna(method='ffill')

# Check the first few rows to understand the structure
print(df.head())

# Plot training and evaluation metrics

# 1. Plot rollout/ep_rew_mean (episode reward) over time (or timesteps)
plt.figure(figsize=(10, 6))
plt.plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], label='Episode Reward (Mean)', color='blue')
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('Episode Reward Over Time')
plt.grid()
plt.legend()
plt.show()

# 2. Plot eval/mean_reward (mean reward during evaluation) over time
plt.figure(figsize=(10, 6))
plt.plot(df['time/total_timesteps'], df['eval/mean_reward'], label='Evaluation Mean Reward', color='green')
plt.xlabel('Timesteps')
plt.ylabel('Mean Evaluation Reward')
plt.title('Mean Evaluation Reward Over Time')
plt.grid()
plt.legend()
plt.show()

# 3. Plot train/loss (training loss) over time (or timesteps)
plt.figure(figsize=(10, 6))
plt.plot(df['time/total_timesteps'], df['train/loss'], label='Training Loss', color='red')
plt.xlabel('Timesteps')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Time')
plt.grid()
plt.legend()
plt.show()

# 4. Plot train/explained_variance (explained variance) over time
plt.figure(figsize=(10, 6))
plt.plot(df['time/total_timesteps'], df['train/explained_variance'], label='Explained Variance', color='purple')
plt.xlabel('Timesteps')
plt.ylabel('Explained Variance')
plt.title('Explained Variance Over Time')
plt.grid()
plt.legend()
plt.show()