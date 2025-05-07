import pandas as pd
import matplotlib.pyplot as plt

# Load Monitor log file
df = pd.read_csv("results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/monitor_logs/monitor_static.csv",
                 comment="#",
                 header=0,      # Treats the next line as the header
                 )

with open("results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/monitor_logs/monitor_static.csv", "r") as f:
    for _ in range(10):
        print(f.readline())


# Plot Episode Rewards over Time
plt.figure(figsize=(8,5))
plt.plot(df['r'], label="Episode Reward", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.legend()
plt.grid()
plt.show()

# plt.savefig("figures/training_progress.png", dpi=300, bbox_inches="tight")
