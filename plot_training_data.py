import pandas as pd
import matplotlib.pyplot as plt
import os 

# Load the CSV
# df = pd.read_csv("results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/2025-04-09_14-44-46/model_log/progress.csv")

log_dir = "results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/2025-04-30_14-29-36/model_log/"
csv_path = os.path.join(log_dir, "progress.csv")
df = pd.read_csv(csv_path)

# Create figures directory if it doesn't exist
figures_dir = os.path.join(log_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# List of columns to plot
# print(df.columns)
# Index(['time/time_elapsed', 'rollout/ep_len_mean', 'rollout/success_rate',
#        'rollout/ep_rew_mean', 'time/iterations', 'time/fps',
#        'time/total_timesteps', 'train/value_loss', 'train/entropy_loss',
#        'train/policy_gradient_loss', 'train/n_updates', 'train/clip_range',
#        'train/loss', 'train/explained_variance', 'train/clip_fraction',
#        'train/approx_kl', 'train/std', 'train/learning_rate',
#        'eval/mean_reward', 'eval/mean_ep_length', 'eval/success_rate'],
#       dtype='object')


# List of columns to plot
metrics = [
    # 'rollout/ep_len_mean',
    'rollout/success_rate',
    'rollout/ep_rew_mean',
    'eval/mean_reward',
    # 'eval/mean_ep_length',
    'eval/success_rate'
]

# Create a plot for each metric
for metric in metrics:
    df_filtered = df.dropna(subset=[metric])
    plt.figure()
    plt.plot(df_filtered['time/total_timesteps'], df_filtered[metric])
    plt.xlabel('Total Timesteps')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Total Timesteps')
    plt.grid(True)
    plt.tight_layout()
    
    # Generate safe filename
    filename = f"{metric.replace('/', '_')}.png"
    save_path = os.path.join(figures_dir, filename)
    plt.savefig(save_path)
    plt.close()  # Close to avoid overlap in multiple plots

plt.show()
