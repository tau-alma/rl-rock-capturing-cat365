'''
This script uses stable-baselines3 to train a policy that can balance an inverted pendulum.

usage: cartpole.py [-h] [--out-dir OUT_DIR] [--load LOAD] [--train]
                   [--observation-space {scalar,visual}]

optional arguments:
  -h, --help            show this help message and exit
  --out-dir OUT_DIR     Path to directory where results from training is saved
  --load LOAD           Path to directory where trained policy is saved
  --train               Whether to train or evaluate. Default is evaluate.
  --observation-space {scalar,visual}
                        What observation space to use. Camera or scalar
'''

import os
import argparse

import gymnasium

from agxPythonModules.agxGym.envs.cartpole_env import CartPoleEnv
from agxPythonModules.agxGym.agx_env import make_env, evaluate_env
from agxPythonModules.agxGym.utils import get_name_from_hp, KeyBoardListenerWrapper
from agxPythonModules.agxGym.baselines_utils import linear_schedule, SmallCNNFeatureExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="results/cartpole", help="Path to directory where results from training is saved")
    ap.add_argument("--load", default=None, type=str, help="Path to directory where trained policy is saved")
    ap.add_argument("--train", default=False, action="store_true", help="Whether to train or evaluate. Default is evaluate.")
    ap.add_argument("--observation-space", default="scalar", choices=["scalar", "visual"], help="What observation space to use. Camera or scalar")
    args = vars(ap.parse_args())

    learning_rate = 2e-4
    batch_size = 256
    epochs = 5
    entropy_coef = 1e-5
    update_interval = 2048

    layers = 2
    hidden_units = 64

    pi_vf = [hidden_units for _ in range(layers)]
    net_arch = dict(pi=pi_vf, vf=pi_vf)
    features_dim = 64

    render_mode = None
    policy_arch = "MlpPolicy"
    policy_kwargs = {"net_arch": net_arch}
    visual_observation_space = False
    if args["observation_space"] == "visual":
        visual_observation_space = True
        policy_arch = "CnnPolicy"
        render_mode = "rgb_array"
        policy_kwargs["features_extractor_class"] = SmallCNNFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(features_dim=features_dim)

    out_dir = get_name_from_hp(
        args["out_dir"],
        o_space="visual" if visual_observation_space else "scalar",
        lr=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        entropy_coef=entropy_coef,
        update_interval=update_interval,
        layers=layers,
        hu=hidden_units
    )

    if args["train"]:
        venv = DummyVecEnv(([make_env(CartPoleEnv, render_mode=render_mode, headless=True, visual_observation_space=visual_observation_space)]))
        if visual_observation_space:
            venv = VecTransposeImage(venv)
            venv = VecFrameStack(venv, n_stack=2)
        eval_callback = EvalCallback(
            venv,
            n_eval_episodes=10,
            best_model_save_path=f"{out_dir}/best",
            log_path=out_dir,
            eval_freq=5000,
            deterministic=True,
            render=False)
    else:
        if render_mode is None:
            render_mode = "human"
        venv = DummyVecEnv([make_env(CartPoleEnv, wrappers=[KeyBoardListenerWrapper], render_mode=render_mode, headless=False, visual_observation_space=visual_observation_space)])
        if visual_observation_space:
            venv = VecTransposeImage(venv)
            venv = VecFrameStack(venv, n_stack=2)

    if args["load"]:
        model = PPO.load(args["load"], venv)
    else:
        model = PPO(
            policy_arch,
            venv,
            policy_kwargs=policy_kwargs,
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
            total_timesteps=3e5,
            callback=eval_callback
        )
        model.save(os.path.join(out_dir, "last"))
        venv.reset()
        venv.close()
    else:
        evaluate_env(venv, model.predict, num_episodes=10)


if __name__ == "__main__":
    main()
