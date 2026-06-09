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
        if metadata.version("rsl-rl-lib") != "3.1.1":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==3.1.1'."
    ) from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# hrp2_env_gs.py が HRP2 にも対応できる汎用的なクラスを含んでいると仮定
# 必要に応じて rl_env_gs_tr.py の RLEnvGenesis を使うように変更してください
from hrp2_env_gs import BP000EnvGenesis as RL_Env


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        ## runner
        "class_name": "OnPolicyRunner",
        # -- general
        "num_steps_per_env": 48,
        "max_iterations": max_iterations,
        "seed": 1,
        # -- observations
        # "obs_groups": {"policy": ["policy"], "critic": ["policy", "privileged"]} # maps observation groups to types. See `vec_env.py` for more information
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy"],
        },  # maps observation groups to types. See `vec_env.py` for more information
        # -- logging parameters
        "save_interval": 100,
        "experiment_name": exp_name,
        "run_name": "",
        # -- logging writer
        "logger": "tensorboard",  # tensorboard, neptune, wandb #not in v2.2.4
        "neptune_project": "legged_gym",  # not in v2.2.4
        "wandb_project": "legged_gym",  # not in v2.2.4
        ##
        "empirical_normalization": None,
        ###
        "algorithm": {
            "class_name": "PPO",
            # -- training
            "learning_rate": 0.001,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            # -- value function
            "value_loss_coef": 1.0,
            "clip_param": 0.2,
            "use_clipped_value_loss": True,
            # -- surrogate loss
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
            # -- miscellaneous
            "normalize_advantage_per_mini_batch": False,  # not in v2.2.4
        },
        "init_member_classes": {},
        ###
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_obs_normalization": False,  # not in v2.2.4
            "critic_obs_normalization": False,  # not in v2.2.4
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",  # 'scalar' or 'log' #not in v2.2.4
        },
    }
    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 30, # ★変更: 全身自由度(12脚+4体幹頭+14腕)
        # joint/link names
        # HRP2_genesis.urdf 用の関節名に変更
        "default_joint_angles": {  # [rad]
            # --- 下半身 (12) ---
            "RLEG_JOINT0": 0.0,   # R_HIP_Y
            "RLEG_JOINT1": 0.0,   # R_HIP_R
            "RLEG_JOINT2": -0.4,  # R_HIP_P
            "RLEG_JOINT3": 0.8,   # R_KNEE_P
            "RLEG_JOINT4": -0.4,  # R_ANKLE_P
            "RLEG_JOINT5": 0.0,   # R_ANKLE_R
            "LLEG_JOINT0": 0.0,   # L_HIP_Y
            "LLEG_JOINT1": 0.0,   # L_HIP_R
            "LLEG_JOINT2": -0.4,  # L_HIP_P
            "LLEG_JOINT3": 0.8,   # L_KNEE_P
            "LLEG_JOINT4": -0.4,  # L_ANKLE_P
            "LLEG_JOINT5": 0.0,   # L_ANKLE_R
            
            # --- 体幹・頭部 (4) ---
            "CHEST_JOINT0": 0.0,  # Waist Yaw
            "CHEST_JOINT1": 0.0,  # Waist Pitch
            "HEAD_JOINT0":  0.0,  # Head Yaw
            "HEAD_JOINT1":  0.0,  # Head Pitch
            
            # --- 右腕 (7) ---
            "RARM_JOINT0": 0.0,   # Shoulder P
            "RARM_JOINT1": 0.0,   # Shoulder R
            "RARM_JOINT2": 0.0,   # Shoulder Y
            "RARM_JOINT3": 0.0,   # Elbow P
            "RARM_JOINT4": 0.0,   # Wrist Y
            "RARM_JOINT5": 0.0,   # Wrist P
            "RARM_JOINT6": 0.0,   # Wrist R
            
            # --- 左腕 (7) ---
            "LARM_JOINT0": 0.0,
            "LARM_JOINT1": 0.0,
            "LARM_JOINT2": 0.0,
            "LARM_JOINT3": 0.0,
            "LARM_JOINT4": 0.0,
            "LARM_JOINT5": 0.0,
            "LARM_JOINT6": 0.0,
        },

        "joint_names": [
            # 順序はObservation/Actionと同期します
            "RLEG_JOINT0", "RLEG_JOINT1", "RLEG_JOINT2", "RLEG_JOINT3", "RLEG_JOINT4", "RLEG_JOINT5",
            "LLEG_JOINT0", "LLEG_JOINT1", "LLEG_JOINT2", "LLEG_JOINT3", "LLEG_JOINT4", "LLEG_JOINT5",
            "CHEST_JOINT0", "CHEST_JOINT1",
            "HEAD_JOINT0", "HEAD_JOINT1",
            "RARM_JOINT0", "RARM_JOINT1", "RARM_JOINT2", "RARM_JOINT3", "RARM_JOINT4", "RARM_JOINT5", "RARM_JOINT6",
            "LARM_JOINT0", "LARM_JOINT1", "LARM_JOINT2", "LARM_JOINT3", "LARM_JOINT4", "LARM_JOINT5", "LARM_JOINT6",
        ],
        # PD gains (HRP2は重いので強めに設定)
        "kp": 2000.0,
        "kd": 100.0,
        # termination
        "termination_if_roll_greater_than": 20,  # 転倒判定を少し緩める
        "termination_if_pitch_greater_than": 20,
        # base pose
        # 腰高さの目標値 (ハーフシッティング時)
        "base_init_pos": [0.0, 0.0, 0.71], 
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25, # 学習初期は小さめが安全
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "dt": 0.01,
        "substeps": 10,
        "rotorInertia": 0.1,
        # noise settings
        "base_roll_noise": [0.00, 0.00],
        "base_pitch_noise": [0.00, 0.00],

        # domain randomization
        "domain_rand": {
            "friction": [0.4, 1.1],     # 地面摩擦係数範囲
            "restitution": [0.0, 0.2],  # 地面反発係数範囲
            "kp": [1800.0, 2200.0],     # Pゲイン範囲 +-10%
            "kd": [80.0, 120.0],        # Dゲイン範囲 +-20%
        },
    }
    obs_cfg = {
        # dof_pos(30) + dof_vel(30) + actions(30) + ang_vel(3) + gravity(3) + commands(3) = 99
        "num_obs": 99, 
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.70, # 直立時 ~0.74m
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 8.0,
            "tracking_ang_vel": 4.0,
            "lin_vel_z": -1.0,
            "base_height": -30.0, # 高さ維持に対する罰則を少し弱める
            "action_rate": -0.01,
            "similar_to_default": -0.2, # ★変更: 全身（特に上半身）の姿勢維持を強く推奨
            "episode_len": 0.01,
            "correct_action": 0.01,
            # "min_ankle_height": 1.0,
            "feet_air_time": 0.2, # 歩行を促進
            "torques": -0.000001,
            "feet_stride": 0.1, # 足の前後距離が大きいほど報酬が増える（歩幅を広げる）
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.8, 0.8],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default='hrp2-walking') # 名前変更
    parser.add_argument("-B", "--num_envs", type=int, default=4096) # GPUメモリに合わせて減らす
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--view", action='store_true')

    args = parser.parse_args()

    # URDFパスをHRP-2に変更
    # 現在のディレクトリ(samples)からの相対パス
    robot_path = os.path.join(os.path.dirname(__file__), "hrp2_description/HRP2_genesis.urdf")

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
            raise Exception("file not found {}".format(resume_path))

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
        robot_urdf_path=robot_path,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume is not None:
        runner.load(resume_path)
    env.reset()
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()

"""
# training
python bp000_train.py
"""
