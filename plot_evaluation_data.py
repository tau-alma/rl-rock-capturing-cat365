import numpy as np
import matplotlib.pyplot as plt
import os 

# Load evaluation data
log_dir = "results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/2025-05-06_12-38-50/model_log/eval/"
npz_path = os.path.join(log_dir, "evaluation_data.npz")
data = np.load(npz_path, allow_pickle=True)


# data = np.load("results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/2025-04-09_17-12-05/model_log/eval/evaluation_data.npz", allow_pickle=True)
index_episode = 4  # You can change this to another index if needed

# Create the 'figures' directory if it doesn't exist
figures_dir = os.path.join(log_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# print(data.files)
# Extract data
episode_rewards = data["episode_rewards"]  # shape: (num_episodes,)
episode_rewards_per_timestep = data["episode_rewards_per_timestep"]  # shape: (num_episodes, variable_length) 
episode_observations = data["episode_observations"]  # shape: (num_episodes, variable_length, obs_dim)
episode_info = data["episode_info"]  # This contains the 'info' dictionaries


# --- 1. Cumulative reward per episode ---
plt.figure()
plt.plot(episode_rewards, marker='o')
plt.title("Cumulative Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.tight_layout()

# Save the figure
save_path = os.path.join(figures_dir, "cumulative_reward_per_episode.png")
plt.savefig(save_path)
plt.close()

# --- 2. Reward per timestep for first few episodes ---
plt.figure()
for i in range(min(5, len(episode_rewards_per_timestep))):  # Plot first 5 episodes max
    plt.plot(episode_rewards_per_timestep[i], label=f"Episode {i+1}")
plt.title("Reward per Timestep")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, "reward_per_timestep.png")
plt.savefig(save_path)
plt.close()

# --- 2.2 Cumulative Reward per timestep for first few episodes ---
plt.figure()
for i in range(min(5, len(episode_rewards_per_timestep))):  # Plot first 5 episodes max
    cumulative_reward = np.cumsum(episode_rewards_per_timestep[i])
    plt.plot(cumulative_reward, label=f"Episode {i+1}")
plt.title("Cumulative Reward per Timestep")
plt.xlabel("Timestep")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, "cumulative_reward_per_timestep.png")
plt.savefig(save_path)
plt.close()


# --- 3. Observation variables over time (for 1 episode) ---
# Extract data for the first episode
rock_positions_x = []
bucket_positions_x = []
target_positions_x = []

rock_positions_z = []
bucket_positions_z = []
target_positions_z = []

chassie_rotation_x = []
chassie_rotation_y = []

arm_prismatic_angles = []
arm_prismatic_speeds = []
arm_prismatic_forces = []

stick_prismatic_angles = []
stick_prismatic_speeds = []
stick_prismatic_forces = []

bucket_prismatic_angles = []
bucket_prismatic_speeds = []
bucket_prismatic_forces = []

reward_rock_target_x_axis = []
reward_rock_target_z_axis = []
reward_rock_bucket_x_axis = []
reward_euler_ang_chassie_body = []
reward_control_input = []
reward_smoothing_control_input = []
reward_terminal_condition = []

condition_rock_target_x_axis = []      
condition_rock_target_z_axis = []
condition_rock_bucket_x_axis = []
condition_euler_ang_chassie_body = []
condition_terminal_condition = []

action_arm = []
action_stick = []
action_bucket = []

# Loop through the timesteps in the first episode
for timestep_info in episode_info[index_episode]:
    # print(len(timestep_info))
    # print(len(episode_info))
    timestep_data = timestep_info[0]
    
    # print(type(timestep_data))
    # print(len(timestep_data))
    
    # Rock, Bucket, and Target Positions
    rock_positions_x.append(timestep_data.get("rock_position_x", np.nan))
    bucket_positions_x.append(timestep_data.get("bucket_position_x", np.nan))
    target_positions_x.append(timestep_data.get("target_position_x", np.nan))
    
    rock_positions_z.append(timestep_data.get("rock_position_z", np.nan))
    bucket_positions_z.append(timestep_data.get("bucket_position_z", np.nan))
    target_positions_z.append(timestep_data.get("target_position_z", np.nan))
    
    # Euler Angles (Chassie Rotation)
    chassie_rotation_x.append(timestep_data.get("chassie_rotation_x", np.nan))
    chassie_rotation_y.append(timestep_data.get("chassie_rotation_y", np.nan))
    
    # Joint Angles, Speeds, and Forces
    arm_prismatic_angles.append(timestep_data.get("arm_prismatic_angle", np.nan))
    arm_prismatic_speeds.append(timestep_data.get("arm_prismatic_speed", np.nan))
    arm_prismatic_forces.append(timestep_data.get("arm_prismatic_force", np.nan))
    
    stick_prismatic_angles.append(timestep_data.get("stick_prismatic_angle", np.nan))
    stick_prismatic_speeds.append(timestep_data.get("stick_prismatic_speed", np.nan))
    stick_prismatic_forces.append(timestep_data.get("stick_prismatic_force", np.nan))
    
    bucket_prismatic_angles.append(timestep_data.get("bucket_prismatic_angle", np.nan))
    bucket_prismatic_speeds.append(timestep_data.get("bucket_prismatic_speed", np.nan))
    bucket_prismatic_forces.append(timestep_data.get("bucket_prismatic_force", np.nan))
    
    reward_rock_target_x_axis.append(timestep_data.get("reward_rock_target_x_axis", np.nan))
    reward_rock_target_z_axis.append(timestep_data.get("reward_rock_target_z_axis", np.nan))
    reward_rock_bucket_x_axis.append(timestep_data.get("reward_rock_bucket_x_axis", np.nan))
    reward_euler_ang_chassie_body.append(timestep_data.get("reward_euler_ang_chassie_body", np.nan))
    reward_control_input.append(timestep_data.get("reward_control_input", np.nan))
    reward_smoothing_control_input.append(timestep_data.get("reward_smoothing_control_input", np.nan))
    reward_terminal_condition.append(timestep_data.get("reward_terminal_condition", np.nan))
    
    condition_rock_target_x_axis.append(timestep_data.get("condition_rock_target_x_axis", np.nan))
    condition_rock_target_z_axis.append(timestep_data.get("condition_rock_target_z_axis", np.nan))
    condition_rock_bucket_x_axis.append(timestep_data.get("condition_rock_bucket_x_axis", np.nan))
    condition_euler_ang_chassie_body.append(timestep_data.get("condition_euler_ang_chassie_body", np.nan))
    condition_terminal_condition.append(timestep_data.get("condition_terminal_condition", np.nan)) 
    
    action_arm.append(timestep_data.get("action_arm", np.nan))
    action_stick.append(timestep_data.get("action_stick", np.nan))
    action_bucket.append(timestep_data.get("action_bucket", np.nan))
    

# Convert to numpy arrays for easier plotting
rock_positions_x = np.array(rock_positions_x)
bucket_positions_x = np.array(bucket_positions_x)
target_positions_x = np.array(target_positions_x)

chassie_rotation_x = np.array(chassie_rotation_x)
chassie_rotation_y = np.array(chassie_rotation_y)

arm_prismatic_angles = np.array(arm_prismatic_angles)
arm_prismatic_speeds = np.array(arm_prismatic_speeds)
arm_prismatic_forces = np.array(arm_prismatic_forces)

stick_prismatic_angles = np.array(stick_prismatic_angles)
stick_prismatic_speeds = np.array(stick_prismatic_speeds)
stick_prismatic_forces = np.array(stick_prismatic_forces)

bucket_prismatic_angles = np.array(bucket_prismatic_angles)
bucket_prismatic_speeds = np.array(bucket_prismatic_speeds)
bucket_prismatic_forces = np.array(bucket_prismatic_forces)

reward_rock_target_x_axis = np.array(reward_rock_target_x_axis)
reward_rock_target_z_axis = np.array(reward_rock_target_z_axis)
reward_rock_bucket_x_axis = np.array(reward_rock_bucket_x_axis)
reward_euler_ang_chassie_body = np.array(reward_euler_ang_chassie_body)
reward_control_input = np.array(reward_control_input)
reward_smoothing_control_input = np.array(reward_smoothing_control_input)
reward_terminal_condition = np.array(reward_terminal_condition)

condition_rock_target_x_axis = np.array(condition_rock_target_x_axis)
condition_rock_target_z_axis = np.array(condition_rock_target_z_axis)
condition_rock_bucket_x_axis = np.array(condition_rock_bucket_x_axis)
condition_euler_ang_chassie_body = np.array(condition_euler_ang_chassie_body)
condition_terminal_condition = np.array(condition_terminal_condition)

action_arm = np.array(action_arm)
action_stick = np.array(action_stick)
action_bucket = np.array(action_bucket)

# Plotting

# --- 1. Rock, Bucket, and Target Position X ---
plt.figure(figsize=(10, 6))
plt.plot(rock_positions_x, label="Rock Position X")
plt.plot(bucket_positions_x, label="Bucket Position X")
plt.plot(target_positions_x, label="Target Position X")
plt.title(f"Rock, Bucket, and Target Position X per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Position X")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"position_x_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(rock_positions_z, label="Rock Position Z")
plt.plot(bucket_positions_z, label="Bucket Position Z")
plt.plot(target_positions_z, label="Target Position Z")
plt.title(f"Rock, Bucket, and Target Position Z per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Position Z")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"position_z_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# # --- 2. Euler Angles (Chassie Rotation) ---
plt.figure(figsize=(10, 6))
plt.plot(chassie_rotation_x, label="Under Carriage Body Rotation X")
plt.plot(chassie_rotation_y, label="Under Carriage Body Rotation Y")
plt.title(f"Euler Angles (Under Carriage Body Rotation) per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Rotation Angle (Radians)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"under_carriage_body_rotation_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 3. Joint Angles ---
plt.figure(figsize=(10, 6))
plt.plot(arm_prismatic_angles, label="Arm Prismatic Angle")
plt.plot(stick_prismatic_angles, label="Stick Prismatic Angle")
plt.plot(bucket_prismatic_angles, label="Bucket Prismatic Angle")
plt.title(f"Joint Angles per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Angle (Radians)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"joint_angles_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 4. Joint Speeds ---
plt.figure(figsize=(10, 6))
plt.plot(arm_prismatic_speeds, label="Arm Prismatic Speed")
plt.plot(stick_prismatic_speeds, label="Stick Prismatic Speed")
plt.plot(bucket_prismatic_speeds, label="Bucket Prismatic Speed")
plt.title(f"Joint Speeds (Observation) per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"joint_speeds_observation_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(action_arm, label="Action Arm")
plt.plot(action_stick, label="Action Stick")
plt.plot(action_bucket, label="Action Bucket")
plt.title(f"Joint Speeds (Control Input) per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"joint_speeds_control_input_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 5. Joint Forces ---
plt.figure(figsize=(10, 6))
plt.plot(arm_prismatic_forces, label="Arm Prismatic Force")
plt.plot(stick_prismatic_forces, label="Stick Prismatic Force")
plt.plot(bucket_prismatic_forces, label="Bucket Prismatic Force")
plt.title(f"Joint Forces per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Force (N)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"joint_forces_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# # dt = 1/60  # seconds between timesteps (adjust this to your env’s frame rate)

# # # --- 4 and 5 Compute power per timestep ---
# # arm_power = np.array(arm_prismatic_forces) * np.array(arm_prismatic_speeds)
# # stick_power = np.array(stick_prismatic_forces) * np.array(stick_prismatic_speeds)
# # bucket_power = np.array(bucket_prismatic_forces) * np.array(bucket_prismatic_speeds)

# # # --- Compute cumulative energy (Joules) ---
# # arm_energy = np.cumsum(arm_power) * dt
# # stick_energy = np.cumsum(stick_power) * dt
# # bucket_energy = np.cumsum(bucket_power) * dt

# # print(f"Total energy used by Arm: {arm_energy[-1]:.2f} J")
# # print(f"Total energy used by Stick: {stick_energy[-1]:.2f} J")
# # print(f"Total energy used by Bucket: {bucket_energy[-1]:.2f} J")

# # # --- Plot energy over time ---
# # plt.figure(figsize=(10, 6))
# # plt.plot(arm_energy, label="Arm Prismatic Energy")
# # plt.plot(stick_energy, label="Stick Prismatic Energy")
# # plt.plot(bucket_energy, label="Bucket Prismatic Energy")
# # plt.title("Cumulative Joint Energy per Timestep (Episode 1)")
# # plt.xlabel("Timestep")
# # plt.ylabel("Energy (Joules)")
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()



# --- 6. Reward for Rock Target X Axis ---
plt.figure(figsize=(10, 6))
plt.plot(reward_rock_target_x_axis, label="Reward Rock Target X Axis")
plt.title(f"Reward for Rock Target X Axis per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"reward_rock_target_x_axis_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()


# --- 7. Reward for Rock Target Z Axis ---
plt.figure(figsize=(10, 6))
plt.plot(reward_rock_target_z_axis, label="Reward Rock Target Z Axis")
plt.title(f"Reward for Rock Target Z Axis per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"reward_rock_target_z_axis_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()


# --- 8. Reward for Rock Bucket X Axis ---
plt.figure(figsize=(10, 6))
plt.plot(reward_rock_bucket_x_axis, label="Reward Rock Bucket X Axis")
plt.title(f"Reward for Rock Bucket X Axis per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"reward_rock_bucket_x_axis_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 9. Reward for Euler Angle Chassie Body ---
plt.figure(figsize=(10, 6))
plt.plot(reward_euler_ang_chassie_body, label="Reward Euler Angle Under Carriage Body")
plt.title(f"Reward for Euler Angle of Under Carriage Body per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"reward_euler_ang_under_carriage_body_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# ---10. Reward for Control Input ---
plt.figure(figsize=(10, 6))
plt.plot(reward_control_input, label="Reward Control Input")
plt.title(f"Reward for Control Input per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"reward_control_input_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 11. Reward for Smoothing Control Input ---
plt.figure(figsize=(10, 6))
plt.plot(reward_smoothing_control_input, label="Reward Smoothing Control Input")
plt.title(f"Reward for Smoothing Control Input per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"reward_smoothing_control_input_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 12. Reward for Terminal Condition ---
plt.figure(figsize=(10, 6))
plt.plot(reward_terminal_condition, label="Reward Terminal Condition")
plt.title(f"Reward for Terminal Condition per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"reward_terminal_condition_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 13. Condition Rock Target X Axis ---
plt.figure(figsize=(10, 6))
plt.plot(condition_rock_target_x_axis, label="Condition Rock Target X Axis")
plt.title(f"Condition: Rock Target X Axis (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Condition (0 or 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"condition_rock_target_x_axis_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 14. Condition Rock Target Z Axis ---
plt.figure(figsize=(10, 6))
plt.plot(condition_rock_target_z_axis, label="Condition Rock Target Z Axis")
plt.title(f"Condition: Rock Target Z Axis (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Condition (0 or 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"condition_rock_target_z_axis_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 15. Condition Rock Bucket X Axis ---
plt.figure(figsize=(10, 6))
plt.plot(condition_rock_bucket_x_axis, label="Condition Rock Bucket X Axis")
plt.title(f"Condition: Rock Bucket X Axis (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Condition (0 or 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"condition_rock_bucket_x_axis_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 16. Condition Euler Angle Chassie Body ---
plt.figure(figsize=(10, 6))
plt.plot(condition_euler_ang_chassie_body, label="Condition Euler Angle Under Carriage Body")
plt.title(f"Condition: Euler Angle of Under Carriage Body (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Condition (0 or 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"condition_euler_ang_under_carriage_body_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# --- 17. Condition Terminal Condition ---
plt.figure(figsize=(10, 6))
plt.plot(condition_terminal_condition, label="Condition Terminal Condition")
plt.title(f"Condition: Terminal Condition (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Condition (0 or 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"condition_terminal_condition_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(reward_rock_target_x_axis, label="Reward Rock Target X Axis")
plt.plot(reward_rock_target_z_axis, label="Reward Rock Target Z Axis")
plt.plot(reward_rock_bucket_x_axis, label="Reward Rock Bucket X Axis")
plt.plot(reward_euler_ang_chassie_body, label="Reward Euler Angle Under Carriage Body")
plt.plot(reward_control_input, label="Reward Control Input")
plt.plot(reward_smoothing_control_input, label="Reward Smoothing Control Input")
plt.plot(reward_terminal_condition, label="Reward Terminal Condition")
plt.title(f"All Reward Components per Timestep (Episode {index_episode + 1})")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(figures_dir, f"all_reward_components_episode_{index_episode + 1}.png")
plt.savefig(save_path)
plt.close()

# # Show all plots
# # plt.show()

