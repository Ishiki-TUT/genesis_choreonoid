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

from bp000_env_gs import BP000EnvGenesis as RL_Env


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
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "R_HIP_Y": 0.0,
            "R_HIP_R": 0.0,
            "R_HIP_P": -0.8,
            "R_KNEE": 1.6,
            "R_ANKLE_P": -0.8,
            "R_ANKLE_R": 0.0,
            "L_HIP_Y": 0.0,
            "L_HIP_R": 0.0,
            "L_HIP_P": -0.8,
            "L_KNEE": 1.6,
            "L_ANKLE_P": -0.8,
            "L_ANKLE_R": 0.0,
        },
        "joint_names": [
            "R_HIP_Y",
            "R_HIP_R",
            "R_HIP_P",
            "R_KNEE",
            "R_ANKLE_P",
            "R_ANKLE_R",
            "L_HIP_Y",
            "L_HIP_R",
            "L_HIP_P",
            "L_KNEE",
            "L_ANKLE_P",
            "L_ANKLE_R",
        ],
        # PD
        "kp": 2000.0,  # 2000.0,
        "kd": 50.0,   # 50.0
        # "kp": 2500.0,
        # "kd": 700.0,
        # termination
        "termination_if_roll_greater_than": 10,  # 10 degree
        "termination_if_pitch_greater_than": 10, # 10 degree
        # base pose
        "base_init_pos": [0.0, 0.0, 0.64], # [0.0, 0.0, 0.64]
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 1.00, # 0.25
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "dt": 0.01,
        "substeps": 10,
        "rotorInertia": 0.1,
        # noise settings
        # "base_roll_noise": [-0.00, 0.00],  # rad (0.5 deg)
        # "base_pitch_noise": [-0.00, 0.00], # rad (0.5 deg)
        "base_roll_noise": [-0.0087, 0.0087],  # rad (0.5 deg)
        "base_pitch_noise": [-0.0087, 0.0087], # rad (0.5 deg)

        # domain randomization
        "domain_rand": {
            "friction": [0.4, 1.1],         # 地面摩擦係数範囲
            "restitution": [0.0, 0.2],      # 地面反発係数範囲
            "kp": [1800.0, 2200.0],       # Pゲイン範囲 +-10%
            "kd": [25.0, 75.0],          # Dゲイン範囲 +-50%
        },
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
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 8.0, # 2.0
            "tracking_ang_vel": 1.0, # 0.4
            "lin_vel_z": -2.0,
            "base_height": -10.0, # -50.0
            "action_rate": -0.01,  # -0.005
            "similar_to_default": -0.1,
            # "effort": 0.01,
            # "tracking_error": -0.1,
            "episode_len": 0.01, # 0.01
            "correct_action": 0.05, # 0.01
            # "min_ankle_height": 1.0, #10.0
            "plus_watt": 0.05, # -0.01
            "ankle_regularization": -0.001,
            "feet_stride": 0.0, # 1.0
            "hip_pitch_motion": 0.2, # 0.1
            "feet_alternating_pos": 2.0, # 5.0
            # "feet_pos_symmetry": -1.0,
            "feet_air_time": 3.0,      # 足を浮かせることを強く推奨
            "torques": -0.00001,        # 無駄な力みを罰する
        },
    }

    # reward_cfg = {
    #     "tracking_sigma": 0.25,
    #     "base_height_target": 0.64,
    #     "feet_height_target": 0.075,
    #     "reward_scales": {
    #         "tracking_lin_vel": 2.0,
    #         "tracking_ang_vel": 1.0,
    #         "lin_vel_z": -1.0,
    #         "base_height": -50.0,
    #         "action_rate": -0.005,
    #         "similar_to_default": -0.1,
    #         # "effort": 0.01,
    #         # "tracking_error": -0.1,
    #         "episode_len": 0.01,
    #         "correct_action": 0.01,
    #         "min_ankle_height": 1,
    #     },
    # }

    # reward_cfg = {
    #         # 追従の許容誤差 (Expリワード用)
    #         "tracking_sigma": 0.25, 
    #         "base_height_target": 0.64, # ロボットの脚長に合わせて調整
    #         "feet_height_target": 0.075,
            
    #         "reward_scales": {
    #             # --- 1. 基本動作 (Main) ---
    #             "tracking_lin_vel": 2.0,   # 前に進む意志 (最優先)
    #             "tracking_ang_vel": 1.0,   # 回転追従

    #             # --- 2. 姿勢の矯正 (Posture) ---
    #             # -50.0 は強すぎました。Exp形式にするか、係数を下げます。
    #             # ここでは扱いやすいExp形式(ガウシアン)を推奨します。
    #             # 関数がない場合は -2.0 程度に弱めてください。
    #             "base_height": 1.0,        # 以前の -50.0 (Penalty) -> +1.0 (Reward/Exp) へ変更推奨
    #             "orientation": 1.0,        # 重力方向の維持 (転倒防止)

    #             # --- 3. 人間らしさの核 (Gait Quality) ---
    #             # ★これが最重要です。足を「運ぶ」動作を作ります。
    #             "feet_air_time": 2.0,      # 足を長く浮かせる -> 歩幅が伸びる
                
    #             # --- 4. 省エネ・滑らかさ (Smoothness) ---
    #             # CSVのトルク飽和(200/-200)を防ぐため、ここを強化します。
    #             "torques": -0.0002,        # 無駄な力みを罰する (必須)
    #             "action_rate": -0.01,      # -0.005 -> -0.01 に倍増して振動抑制
    #             "dof_vel": -0.001,         # 関節を早く動かしすぎるのを防ぐ
    #             "dof_acc": -2.5e-7,        # 急加速(ガタツキ)を防ぐ

    #             # --- 5. 形状の補正 ---
    #             "feet_parallel": -0.5,     # ガニ股防止
    #             # "dof_pos_limits": -1.0,    # 関節可動域の限界に当たるのを防ぐ

    #             # --- 削除/変更した項目 ---
    #             # "lin_vel_z": -1.0,       # 上下動抑制は base_height があれば弱くてOK (-0.5程度へ)
    #             # "min_ankle_height": 0.0, # これが足を上げすぎて不安定にさせている可能性大 -> 削除推奨
    #         },
    #     }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5], # [0.5, 0.5]
        # "lin_vel_x_range": [0.0, 1.0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default='bp000-walking')
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--view", action='store_true')

    args = parser.parse_args()

    ## set robot path fix collisiton 
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # /userdir
    robot_path = os.path.join(ROOT, "userdir", "humanoid_research_k", "robots", "kawada_base.simple_collision.urdf")

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
