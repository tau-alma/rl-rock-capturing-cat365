# RL Rock Capturing CAT365


# Automatic Rock Capturing using excavator CAT365 based on Reinforcement Learning (RL)

This project implements Reinforcement Learning (RL) for a rock capturing task using the **Excavator CAT365** model in the **AGX Dynamics** simulator. The goal is to develop and evaluate control policies capable of autonomously operating the excavator to capture rocks.

## Project Overview

- **Simulator**: [AGX Dynamics](https://www.algoryx.se/agx-dynamics/) by Algoryx Simulation AB
- **Excavator Model**: Excavator CAT365
- **Control Algorithm**: Reinforcement Learning (Stable-Baselines3)
- **Task**: Automatic rock capturing
- **Environment**: Custom AGX-Gym environment with excavator, terrain and rock assets

## Repository Structure

```text
project/
├── agxGym/                             # Main gym-style environment package
│   ├── agx_env.py                      # AGX environment setup
│   ├── baselines_utils.py              # Utilities for RL training and evaluation
│   ├── envs/                           # Custom environments and models
│   │   ├── cartpole_env.py             # CartPole environment
│   │   ├── excavator_env.py            # Excavator environment 
│   │   ├── pushing_robot_env.py        # Pushing robot environment
│   │   ├── wheelloader_env.py          # Wheel loader environment
│   │   └── models/                     # AGX simulation models (excavator, terrain, etc.)
├── environment_rlagx.yml               # Conda environment definition file
├── excavator365_RockCapturing.py       # Main script to train or test the agent
├── run_env.py                          # Script to manually run the environment
├── media/                              # Sample video and media assets for README
├── Others/                             # Miscellaneous and legacy scripts
├── plot_training_data.py               # Plot training metrics from log files
├── plot_evaluation_data.py             # Plot test results
├── results/                            # Output directory for logs and model checkpoints
├── Rock_Bucket_initial_conditions.ods  # Initial configuration for the bucket, stick, arm
└── README.md                           # Project overview and instructions
```

## Features

- AGX simulation integration with RL pipeline
- Reward shaping for efficient rock capturing
- Logging with TensorBoard and performance monitoring
- Support for training reproducibility and evaluation

## System Requirements

The following configuration was used to develop and test the project:
- Operating System: Ubuntu 22.04 (Linux)
- CPU: Intel Xeon E5-1650 v2 @ 3.50GHz
- GPU Model: NVIDIA GeForce RTX 4070
- Python Version: 3.10
- Conda: Yes (using conda environment for package management) 
- AGX Dynamics (version: 2.39.0.0)
- Libraries: Gymnasium, Stable-Baselines3, NumPy, PyTorch, etc.

## Getting Started

Follow these steps to set up the project on your machine.

### 1. Clone the repository

```bash
git clone https://git.algoryx.se/algoryx/external/xscave/rl-rock-capturing-cat365.git
```

### 2. Create and activate a Conda environment

You can install dependencies via:

```bash
conda env create -f environment.yml
conda activate rlagx
```

### 3. Test environment manually via keyboard input
You can manually operate the excavator and test the environment using:

```bash
python run_env.py
```

### 4. Train using PPO

You can train a PPO againt using:

```bash
python excavator365_RockCapturing.py --train
```

### 5. Monitor Training with TensorBoard

To visualize the training progress, launch TensorBoard with the log directory of your specific training run:

```bash
tensorboard --logdir <path_to_training_run>
```
For example:

```bash
tensorboard --logdir results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/
```

### 6. Test the trained agent

To test the trained agent, run the following command, replacing `<path_to_model>` with the full path to your saved `.zip` model:

```bash
python excavator365_RockCapturing.py --load <path_to_model>
```

For example:

```bash
python excavator365_RockCapturing.py --load results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/2025-05-06_16-47-42/model_log/best_model.zip
```

### 7. Plot Training Data

To plot the training progress (e.g., rewards), use the provided plotting script:

```bash
python plot_training_data.py
```

Note: Inside the `plot_training_data.py` script, make sure to set the correct path for the log directory by modifying the `log_dir` variable. For example:

```bash
log_dir = "results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/2025-04-30_14-29-36/model_log/"
```

### 8. Plot Evaluation Results

To visualize the evaluation results of the trained agent (e.g., rewards, observations and control inputs), you can run the following script:

```bash
python plot_evaluation_data.py
```

Note: Before running, make sure to update the `log_dir` inside the `plot_evaluation_data.py` script to point to the correct evaluation result directory. For example:

```bash
log_dir = "results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/2025-05-06_12-38-50/model_log/eval/"
```

## Visuals

You can convert webm video file to a GIF using ffmpeg:

```bash
ffmpeg -i Video_Sample_Rock_Capturing.webm -vf "scale=640:-1:flags=lanczos" -c:v gif Video_Sample_Rock_Capturing.gif
```
Below is a sample video demonstrating the automatic rock capturing task using trained PPO agent:

![Rock Capturing Demo](media/Video_Sample_Rock_Capturing.gif)

## Contributing

We welcome contributions to improve and extend this project! If you'd like to contribute via GitLab, follow these steps:

1. **Clone** the repository to your local machine:

   ```bash
   git clone https://git.algoryx.se/algoryx/external/xscave/rl-rock-capturing-cat365.git
   ```

2. **Create a new branch** for your feature or fix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Develop** your changes locally. Make sure your code is clean, tested, and well-documented.

4. **Commit** your changes with a clear message:

   ```bash
   git add .
   git commit -m "Add: meaningful description of your change"
   ```
5. **Push** the branch to GitLab:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Merge Request (MR)** targeting the main branch.

7. Your MR will be reviewed. Please be responsive to feedback and update your branch as needed.

## Authors and acknowledgment
This project is developed and maintained by:

- **Amirmasoud Molaei**  
  Email: [amirmasoud.molaei@tuni.fi](mailto:amirmasoud.molaei@tuni.fi)

Special thanks to the following contributors:

- Algoryx Simulation for providing the AGX Dynamics simulator
- Open-source community contributors to the Gymnasium and Stable-Baselines3 libraries, whose work was instrumental in the development of this project

We also thank everyone who has contributed to improving this project.

## License
This project is currently closed-source for internal research. Licensing terms will be defined in a future release.

## Project status
This project is part of the research project XSCAVE (https://www.xscave.eu/). We are planning to publish our work in the Automation in Construction journal (https://www.sciencedirect.com/journal/automation-in-construction).