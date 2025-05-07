from tbparse import SummaryReader
import matplotlib.pyplot as plt

log_path = "results/excavator365-RockCapturing/*/2025-04-03_15-33-15/tensorboard_logs/PPO_1"
reader = SummaryReader(log_path, pivot=True)
df = reader.scalars

plt.figure(figsize=(10, 5))
for tag in ["train/loss", "train/policy_entropy"]:  # Adjust based on available tags
    if tag in df["tag"].unique():
        temp_df = df[df["tag"] == tag]
        plt.plot(temp_df["step"], temp_df["value"], label=tag)

plt.xlabel("Timesteps")
plt.ylabel("Loss")
plt.title("Loss & Policy Entropy")
plt.legend()
plt.grid()
plt.show()