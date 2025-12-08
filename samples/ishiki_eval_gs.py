import argparse
import os
import pickle

import numpy as np
import pandas as pd

from importlib import metadata
ver_rsl_rl = metadata.version("rsl-rl-lib")
print(f"Version of rsl-rl-lib : {ver_rsl_rl}")

import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

from bp000_env_gs import BP000EnvGenesis as RLEnv

def save_simple_csv(step_data, obs_data, exp_name, ckpt, torque_data=None, dof_pos_data=None, dof_vel_data=None, action_scale=1.0):
    """CSVファイルにデータを保存する関数"""
    if not step_data:
        print("データがありません")
        return None
    
    # 基本的な辞書形式でデータを整理
    data_dict = {'step': step_data}
    
    # Observationデータ
    obs_array = np.array(obs_data)
    for i in range(obs_array.shape[1]):
        data_dict[f'obs_{i}'] = obs_array[:, i]

    # Torqueデータ（任意）
    if torque_data is not None and len(torque_data) > 0:
        tq_array = np.array(torque_data)
        for i in range(tq_array.shape[1]):
            data_dict[f'torque_{i}'] = tq_array[:, i]
    
    # ★追加: 関節角度 (dof_pos)
    if dof_pos_data is not None and len(dof_pos_data) > 0:
        pos_array = np.array(dof_pos_data)
        for i in range(pos_array.shape[1]):
            data_dict[f'dof_pos_{i}'] = pos_array[:, i]

    # ★追加: 関節角速度 (dof_vel)
    if dof_vel_data is not None and len(dof_vel_data) > 0:
        vel_array = np.array(dof_vel_data)
        for i in range(vel_array.shape[1]):
            data_dict[f'dof_vel_{i}'] = vel_array[:, i]

    df = pd.DataFrame(data_dict)

    
    
    # obs_dataディレクトリを作成
    os.makedirs('obs_data', exist_ok=True)
    csv_filename = f'obs_data/genesis_{exp_name}_ckpt{ckpt}_scale{action_scale}.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"データを保存しました: {csv_filename}")
    print(f"データ形状: {df.shape}")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="bp000-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100, help="データ収集ステップ数")
    parser.add_argument("--action_scale", type=float, default=1.0)
    args = parser.parse_args()

    ## set robot path fix collisiton 
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # /userdir
    robot_path = os.path.join(ROOT, "userdir", "humanoid_research_k", "robots", "kawada_base.simple_collision.urdf")

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    ## override
    env_cfg["episode_length_s"] = 20.0
    command_cfg["lin_vel_x_range"] = [0.5, 0.5]
    env_cfg['base_roll_noise'] = [0,0]
    env_cfg['base_pitch_noise'] = [0,0]
    # env_cfg['kp'] = 10000.0
    # env_cfg['kd'] = 50.0

    env = RLEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        dt=env_cfg['dt'],
        substeps=env_cfg['substeps'],
        show_viewer=True,
        robot_urdf_path=robot_path,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device='cuda')
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device='cuda')

    return env, policy, args

def _read_torques(env):
    """環境から実トルクをnp.ndarrayで取得"""
    try:
        if hasattr(env, "robot") and hasattr(env.robot, "get_dofs_force"):
            if hasattr(env, "motors_dof_idx"):
                t = env.robot.get_dofs_force(env.motors_dof_idx)
            else:
                t = env.robot.get_dofs_force()
            return t.detach().cpu().numpy().astype(np.float32).ravel() if isinstance(t, torch.Tensor) \
                   else np.asarray(t, dtype=np.float32).ravel()
        elif hasattr(env, "dof_force") and env.dof_force is not None:
            return env.dof_force[0].detach().cpu().numpy().astype(np.float32).ravel()
    except Exception as e:
        print(f"warn: torque read failed: {e}")
    return np.zeros(env.env_cfg.get("num_actions", 12), dtype=np.float32)

def _obs_vec(obs):
    # TensorDict or dict → 'policy' を優先
    if isinstance(obs, dict) or hasattr(obs, "get"):
        if "policy" in obs:
            obs = obs["policy"]
        # else:
        #     # 値を平坦化（ほぼ使わない想定）
        #     parts = []
        #     for v in obs.values():
        #         if torch.is_tensor(v):
        #             parts.append(v.view(-1))
        #     if parts:
        #         obs = torch.cat(parts)
    if torch.is_tensor(obs):
        return obs.detach().cpu().numpy().ravel()
    return np.asarray(obs, dtype=np.float32).ravel()

def eval_policy_with_data_collection(env, policy, args):
    step_data = []
    obs_data = []
    torque_data = []
    dof_pos_data = []
    dof_vel_data = []

    obs, _ = env.reset()
    cnt = 0

    current_dof_pos = env.dof_pos[0].cpu().numpy() 
    current_dof_vel = env.dof_vel[0].cpu().numpy()

    step_data.append(cnt)
    obs_data.append(_obs_vec(obs))
    torque_data.append(_read_torques(env))
    dof_pos_data.append(current_dof_pos)
    dof_vel_data.append(current_dof_vel)

    print(f"データ収集開始: {args.steps} ステップ")

    with torch.no_grad():
        for i in range(args.steps):
            actions = policy(obs)
            actions = actions * args.action_scale
            obs, rews, dones, infos = env.step(actions)
            current_dof_pos = env.dof_pos[0].cpu().numpy() 
            current_dof_vel = env.dof_vel[0].cpu().numpy()

            cnt += 1
            step_data.append(cnt)
            obs_data.append(_obs_vec(obs))
            torque_data.append(_read_torques(env))
            dof_pos_data.append(current_dof_pos)
            dof_vel_data.append(current_dof_vel)

            if i % 20 == 0:
                print(f"Step {i+1}/{args.steps}, Total steps: {cnt}")

            if dones.any():
                obs, _ = env.reset()
                cnt += 1
                step_data.append(cnt)
                obs_data.append(_obs_vec(obs))
                torque_data.append(_read_torques(env))
                dof_pos_data.append(current_dof_pos)
                dof_vel_data.append(current_dof_vel)

    print(f"データ収集完了: {len(step_data)} rows")

    # CSVファイルに保存（トルク付き）
    df = save_simple_csv(step_data, obs_data, args.exp_name, args.ckpt, torque_data, dof_pos_data, dof_vel_data, args.action_scale)
    return df

def eval_policy_continuous(env, policy, args):
    """連続評価（データ収集なし）"""
    cnt = 0
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            cnt += 1
            actions = policy(obs)
            actions = actions * args.action_scale  # スケール調整
            obs, rews, dones, infos = env.step(actions)
            print(f"Step: {cnt}")
            print("actions : ", actions)
            print("obs : ", obs["policy"])
            
            
            

if __name__ == "__main__":
    env, policy, args = main()
    
    # データ収集付き評価を実行
    df = eval_policy_with_data_collection(env, policy, args)
    
    # 必要に応じて連続評価も実行
    # eval_policy_continuous(env, policy, args)

"""
# 使用例:
# データ収集付き評価(100ステップ)
python3 ishiki_eval_gs.py -e ishiki-walking-no-vel --ckpt 2000 --steps 100 --action_scale 0.5

# データ収集付き評価(500ステップ)
python3 ishiki_eval_gs.py -e ishiki-walking-no-vel --ckpt 2000 --steps 500 --action_scale 0.5
"""