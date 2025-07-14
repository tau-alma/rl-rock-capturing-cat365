'''
usage: excavator365_RockCapturing.py [-h] [--out-dir OUT_DIR] [--load LOAD] [--train]

optional arguments:
  -h, --help         show this help message and exit
  --out-dir OUT_DIR  Path to directory where results from training is saved
  --load LOAD        Path to directory where agent results is saved
  --train            Whether to train or evaluate. Default is evaluate.
'''

import os
import argparse
import time
import torch
from datetime import datetime

# from agxPythonModules.agxGym.envs.excavator_env import ExcavatorTerrainEnv
from agxGym.envs.excavator_env import ExcavatorTerrainEnv
# from agxPythonModules.agxGym.agx_env import make_env, evaluate_env
from agxGym.agx_env import make_env, evaluate_env, evaluate_env_extra_variables
from agxPythonModules.agxGym.utils import get_name_from_hp, NormalizeObsSpace, KeyBoardListenerWrapper
from agxPythonModules.agxGym.baselines_utils import linear_schedule

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv 
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    start_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", 
                    type=str, 
                    default="results/excavator365-RockCapturing", 
                    help="Path to directory where results from training is saved")
    ap.add_argument("--load", 
                    default=None, 
                    # default='results/excavator365-RockCapturing/hp_lr_0.0003-batch_size_128-epochs_4-entropy_coef_0.0003-update_interval_1024-/best/best_model',
                    type=str, 
                    help="Path to directory where agent results is saved")
    ap.add_argument("--train", 
                    default=False, 
                    action="store_true", 
                    help="Whether to train or evaluate. Default is evaluate.")
    args = vars(ap.parse_args())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 12
    # num_envs = min(os.cpu_count() // 2, 8) 
        
    learning_rate = 3e-4
    batch_size = 128 
    epochs = 4
    entropy_coef = 3e-4
    update_interval = 1024 

    # update_interval = 2048
    # batch_size = 512


    render_mode = None

    layers = 2 # 2, 3, 4, 5                        3
    hidden_units = 128 #128, 256, 512, 1024        512
    pi_vf = [hidden_units for _ in range(layers)]
    net_arch = dict(pi=pi_vf, vf=pi_vf)
    policy_arch = "MlpPolicy"
    policy_kwargs = {"net_arch": net_arch}
    
    out_dir = get_name_from_hp(
        args["out_dir"],
        lr=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        entropy_coef=entropy_coef,
        update_interval=update_interval,
    )
    
    # Get the current date and time in the format YYYY-MM-DD_HH-MM-SS
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"{out_dir}/{current_time}"
    
    monitor_log_dir = f"{out_dir}/monitor_logs"
    eval_monitor_log_dir = f"{out_dir}/monitor_logs/eval"
    checkpoint_log_dir = f"{out_dir}/checkpoint_logs"
    model_log_dir = f"{out_dir}/model_log"
    eval_log_dir = f"{out_dir}/model_log/eval"
    
    # vec_monitor_log_dir = f"{out_dir}/vec_monitor_logs"
    # eval_vec_monitor_log_dir = f"{out_dir}/vec_monitor_logs/eval"
    # tensorboard_log_dir = f"{out_dir}/tensorboard_logs"

    if args["train"]:
        
        # venv = SubprocVecEnv(([make_env(
        #     ExcavatorTerrainEnv,
        #     wrappers=[NormalizeObsSpace],
        #     render_mode=render_mode,
        #     seed=i) for i in range(num_envs)])
        # )
        
        # # Wrap it with VecMonitor for automatic logging
        # venv = VecMonitor(venv, filename=os.path.join(vec_monitor_log_dir, 'monitor.csv'))
        
        # eval_venv = SubprocVecEnv([make_env(
        #     ExcavatorTerrainEnv,
        #     wrappers=[NormalizeObsSpace],
        #     render_mode=render_mode,
        #     seed=num_envs)])
        
        # eval_venv = VecMonitor(eval_venv, filename=os.path.join(eval_vec_monitor_log_dir, 'monitor.csv'))
        
        venv = make_vec_env(env_id=make_env(ExcavatorTerrainEnv, render_mode=render_mode), 
                            n_envs=num_envs, 
                            seed=0,
                            monitor_dir=monitor_log_dir,
                            wrapper_class=NormalizeObsSpace,
                            vec_env_cls=SubprocVecEnv)

        eval_venv = make_vec_env(env_id=make_env(ExcavatorTerrainEnv, render_mode=render_mode), 
                            n_envs=1, 
                            seed=num_envs,
                            monitor_dir=eval_monitor_log_dir,
                            wrapper_class=NormalizeObsSpace,
                            vec_env_cls=SubprocVecEnv)
        
        # Callback function to evaluate the model
        eval_callback = EvalCallback(
            eval_venv,
            n_eval_episodes=10,
            best_model_save_path=model_log_dir,#f"{model_log_dir}/best",
            log_path=eval_log_dir,
            eval_freq=3000, # 36000 // num_envs,
            deterministic=True,
            render=False)
        
        # Save a checkpoint every 3000 time steps
        checkpoint_callback = CheckpointCallback(
            save_freq=3000, # 36000 // num_envs,
            save_path=checkpoint_log_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True)

    else:
        # venv = DummyVecEnv([make_env(
        #     ExcavatorTerrainEnv,
        #     wrappers=[NormalizeObsSpace, KeyBoardListenerWrapper],
        #     render_mode="human",
        #     headless=False)])
        
        # venv = VecMonitor(venv, filename=os.path.join(eval_log_dir, 'monitor.csv'))
        
        venv = make_vec_env(env_id=make_env(ExcavatorTerrainEnv, render_mode="human", headless=False), 
                            n_envs=1, 
                            monitor_dir=eval_log_dir,
                            wrapper_class=NormalizeObsSpace,
                            vec_env_cls=DummyVecEnv)

    if args["load"]:
        model = PPO.load(args["load"], venv, device=device)
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
            # tensorboard_log=tensorboard_log_dir,
            device=device,
        )
        log = configure(folder=model_log_dir, format_strings=["stdout", "csv", "tensorboard"])
        model.set_logger(logger=log)

    if args["train"]:
        model.learn(
            total_timesteps=15e6,
            callback=[eval_callback, checkpoint_callback])
        
        model.save(os.path.join(model_log_dir, "last"))
        venv.reset()
        venv.close()
    else:
        # evaluate_env(venv, 
        #              model.predict, 
        #              num_episodes=5)
        
        os.makedirs(eval_log_dir, exist_ok=True)
        evaluate_env_extra_variables(venv, 
                     model.predict, 
                     num_episodes=5,
                     save_dir=os.path.join(eval_log_dir, "evaluation_data.npz"))
    
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f"Elapsed time: {elapsed_time:.2f} minutes")


if __name__ == "__main__":
    main()
