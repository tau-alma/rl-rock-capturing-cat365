'''
Trains a robot that have torque controlled hinge motors to push a box on a rail as far back as possible.

usage: pushing_robot.py [-h] [--out-dir OUT_DIR] [--load LOAD] [--train]

optional arguments:
  -h, --help         show this help message and exit
  --out-dir OUT_DIR  Path to directory where results from training is saved
  --load LOAD        Path to directory where agent results is saved
  --train            Whether to train or evaluate. Default is evaluate.
'''

import os
import argparse

from agxPythonModules.agxGym.envs.pushing_robot_env import PushingRobotEnv
from agxPythonModules.agxGym.agx_env import make_env, evaluate_env
from agxPythonModules.agxGym.utils import get_name_from_hp, NormalizeObsSpace, KeyBoardListenerWrapper
from agxPythonModules.agxGym.baselines_utils import linear_schedule

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="results/pushing-robot", help="Path to directory where results from training is saved")
    ap.add_argument("--load", default=None, type=str, help="Path to directory where agent results is saved")
    ap.add_argument("--train", default=False, action="store_true", help="Whether to train or evaluate. Default is evaluate.")
    args = vars(ap.parse_args())

    learning_rate = 3e-4
    batch_size = 128
    epochs = 4
    entropy_coef = 3e-4
    update_interval = 1024

    render_mode = None
    policy_arch = "MlpPolicy"

    num_envs = 2

    out_dir = get_name_from_hp(
        args["out_dir"],
        lr=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        entropy_coef=entropy_coef,
        update_interval=update_interval,
    )

    if args["train"]:
        venv = SubprocVecEnv(([make_env(
            PushingRobotEnv,
            wrappers=[NormalizeObsSpace],
            render_mode=render_mode,
            seed=i) for i in range(num_envs)])
        )
        eval_venv = SubprocVecEnv([make_env(
            PushingRobotEnv,
            wrappers=[NormalizeObsSpace],
            render_mode=render_mode,
            seed=num_envs)])

        eval_callback = EvalCallback(
            eval_venv,
            n_eval_episodes=10,
            best_model_save_path=f"{out_dir}/best",
            log_path=out_dir,
            eval_freq=20000,
            deterministic=True,
            render=False)
    else:
        venv = DummyVecEnv([make_env(
            PushingRobotEnv,
            wrappers=[NormalizeObsSpace, KeyBoardListenerWrapper],
            render_mode="human",
            headless=False)])

    if args["load"]:
        model = PPO.load(args["load"], venv)
    else:
        model = PPO(
            policy_arch,
            venv,
            verbose=1,
            learning_rate=linear_schedule(learning_rate),
            batch_size=batch_size,
            n_epochs=epochs,
            ent_coef=entropy_coef,
            n_steps=update_interval,
            tensorboard_log=out_dir,
        )

    if args["train"]:
        model.learn(
            total_timesteps=5e5,
            callback=eval_callback
        )
        model.save(os.path.join(out_dir, "last"))
        venv.reset()
        venv.close()
    else:
        evaluate_env(venv, model.predict)


if __name__ == "__main__":
    main()
