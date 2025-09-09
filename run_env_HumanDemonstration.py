'''
Runs any of the registered environments with 'agx' in the name using a random policy
or heuristic/keyboard-policy if they are implemented in the environment.

Usage:

`python run_env.py -l` - to list all the available environments
`python run_env.py --env name-of-environment` - to choose to run the specified environment.
'''

import gymnasium as gym
from gymnasium import envs
import argparse
import numpy as np
import os
from datetime import datetime, timezone


# from agxPythonModules import agxGym
# from agxPythonModules.agxGym.agx_env import AGXGymEnv
from agxGym.agx_env import AGXGymEnv
#from agxPythonModules.agxGym.utils import ExitException, KeyBoardListenerWrapper
from agxGym.utils import ExitException, KeyBoardListenerWrapper
from agxPythonModules.utils.environment import simulation


def run(args):
    env: AGXGymEnv = gym.make(args['env'], render_mode="human", headless=False)

    def policy(t):
        return env.action_space.sample()

    if args['policy'] == "keyboard":
        policy = env.keyboard_control_policy
    elif args['policy'] == 'heuristic':
        policy = env.heuristic_control_policy

    # Want to be able to interact with the simulation window
    env = KeyBoardListenerWrapper(env)
    
    out_dir = "results/excavator365-RockCapturing/human_operators"
    # Get the current date and time in the format YYYY-MM-DD_HH-MM-SS
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Current time: {current_time}")
    out_dir = f"{out_dir}/{current_time}"
    eval_log_dir = f"{out_dir}/model_log/eval"
    os.makedirs(eval_log_dir, exist_ok=True)
    save_dir = os.path.join(eval_log_dir, "evaluation_data.npz")
    
    all_episode_info = []            # list of info dicts per episode
    all_episode_obs = []         # list of arrays: observations per episode
    
    all_episode_rewards = []     # list of scalars: total reward per episode
    all_episode_rewards_per_timestep = []  # list of arrays: rewards per timestep
    
    try:
        for e in range(args["nr_episodes"]):
            obs, info = env.reset()
            
            if args["journalRecord"]:
                env.start_recording_journal(f"test_{e}.agxJournal", journal_config_path="JournalConfig.json")
            terminal = False
            truncated = False
            tot_reward = 0
            
            episode_reward = 0
            
            episode_obs = []
            episode_info = []  
            episode_rewards = []
            
            while not terminal and not truncated:         
                episode_obs.append(obs)
                
                obs, reward, terminal, truncated, info = env.step(policy(env.unwrapped.sim.getTimeStamp()))
                tot_reward += reward
                
                # Save the info for the current timestep
                episode_info.append(info)
                
                episode_reward += reward
                episode_rewards.append(reward)
            
            episode_obs = [np.squeeze(obs) for obs in episode_obs]
            all_episode_obs.append(np.array(episode_obs))
            all_episode_info.append(episode_info)  # Append info for this episode
            
            all_episode_rewards.append(episode_reward)
            all_episode_rewards_per_timestep.append(np.array(episode_rewards))
                
            print(f"Total reward for episode {e} is {tot_reward}!")
        
        # Save data
        np.savez_compressed(
            save_dir,
            episode_rewards=np.array(all_episode_rewards),  
            episode_rewards_per_timestep=np.array(all_episode_rewards_per_timestep, dtype=object),
            episode_observations=np.array(all_episode_obs, dtype=object),
            episode_info=np.array(all_episode_info, dtype=object)
        )   
        
        print(f"\n Saved evaluation data to {save_dir}")
        
    except ExitException:
        print("Exit application")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="agx-365-terrain-rock-v0")
    ap.add_argument("--policy", type=str, default="keyboard", choices=["random", "heuristic", "keyboard"])
    ap.add_argument("-l", action="store_true", default=False)
    ap.add_argument("--journalRecord", action="store_true", default=False)
    ap.add_argument("--journalConfigPath", default="JournalConfig.json", type=str)
    ap.add_argument("--nr-episodes", type=int, default=10)

    args = vars(ap.parse_args())

    if args["l"]:
        all_envs = envs.registry.items()
        env_ids = [name for name, _ in all_envs if 'agx' in name]
        print("Available AGX environments are: ")
        print(env_ids)
        return

    run(args)


if __name__ == "__main__":
    main()
