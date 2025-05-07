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
    try:
        for e in range(args["nr_episodes"]):
            env.reset()
            if args["journalRecord"]:
                env.start_recording_journal(f"test_{e}.agxJournal", journal_config_path="JournalConfig.json")
            terminal = False
            truncated = False
            tot_reward = 0
            while not terminal and not truncated:
                obs, reward, terminal, truncated, _ = env.step(policy(env.unwrapped.sim.getTimeStamp()))
                tot_reward += reward
            print(f"Total reward for episode {e} is {tot_reward}!")
    except ExitException:
        print("Exit application")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="agx-365-terrain-rock-v0")
    ap.add_argument("--policy", type=str, default="keyboard", choices=["random", "heuristic", "keyboard"])
    ap.add_argument("-l", action="store_true", default=False)
    ap.add_argument("--journalRecord", action="store_true", default=False)
    ap.add_argument("--journalConfigPath", default="JournalConfig.json", type=str)
    ap.add_argument("--nr-episodes", type=int, default=100)

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
