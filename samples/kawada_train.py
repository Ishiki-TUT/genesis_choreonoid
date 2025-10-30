import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from kawada_env_gs import KawadaBaseEnvGenesis as RL_Env

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 48,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            'R_HIP_Y': 0.0,
            'R_HIP_R': 0.0,
            'R_HIP_P': -0.8,
            'R_KNEE': 1.6,
            'R_ANKLE_P': -0.8,
            'R_ANKLE_R': 0.0,
            'L_HIP_Y': 0.0,
            'L_HIP_R': 0.0,
            'L_HIP_P': -0.8,
            'L_KNEE': 1.6,
            'L_ANKLE_P': -0.8,
            'L_ANKLE_R': 0.0
        },
        "joint_names": [
            'R_HIP_Y',
            'R_HIP_R',
            'R_HIP_P',
            'R_KNEE',
            'R_ANKLE_P',
            'R_ANKLE_R',
            'L_HIP_Y',
            'L_HIP_R',
            'L_HIP_P',
            'L_KNEE',
            'L_ANKLE_P',
            'L_ANKLE_R'
        ],
        # PD
        # "kp": 10000.0,
        # "kd": 1000.0,
        # "kp": 80000.0,
        # "kd": 200.0,
        # "kp": 20.0,
        # "kd": 0.5,
        "kp": 2000.0,
        "kd": 500.0,
        # "kp": 2000.0,
        # "kd": 10.0,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.64],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "dt": 0.01,    ## add by IRSL
        "substeps": 5, ## add by IRSL
        "rotorInertia": 0.1, ## add by IRSL
    }
    obs_cfg = {
        "num_obs": 45,  # 3 + 3 + 3 + 12 + 12 + 12
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.64,
        # "base_height_target": 0.70,
        "feet_height_target": 0.075,
        "reward_scales": {
            # "tracking_lin_vel": 1.0,
            # "tracking_ang_vel": 0.2,
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.4,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            # "action_rate": -0.05,
            "similar_to_default": -0.1,
            # "effort": 0.,
            # "tracking_error": -0.1,
            "episode_len" : 0.1,
            "correct_action"  : 0.01,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        # "lin_vel_x_range": [0.0, 1.0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="kawada-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--view", action='store_true')

    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    ## over-write parmeter here


    ## making directory
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.resume is not None:
        resume_path = args.resume
        if not os.path.isfile(resume_path):
            raise Exception('file not found {}'.format(resume_path))

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = RL_Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        dt=env_cfg['dt'],
        substeps=env_cfg['substeps'],
        show_viewer=args.view,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume is not None:
        runner.load(resume_path)
    # env.reset()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/bex24_train.py
"""
