import argparse
import os
import pickle
import pandas as pd
import numpy as np

from importlib import metadata
import torch
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
# from ishiki_env_gs import KawadaBaseEnvGenesis as RL_Env

def save_simple_csv(step_data, obs_data, exp_name, ckpt, torque_data=None, scale=1.0):
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
    
    df = pd.DataFrame(data_dict)
    
    # obs_dataディレクトリを作成
    os.makedirs('obs_data', exist_ok=True)
    csv_filename = f'obs_data/genesis_{exp_name}_ckpt{ckpt}_scale{scale}_fixed_0.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"データを保存しました: {csv_filename}")
    print(f"データ形状: {df.shape}")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="kawada-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100, help="データ収集ステップ数")
    parser.add_argument("--scale", type=int, default=1.0, help="データ収集アクションスケール")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}


    # env_cfg["base_init_pos"] = [0.0, 0.0, 1.0] # 高さに変更 64
    ## override
    # env_cfg["episode_length_s"] = 40.0
    # command_cfg["lin_vel_x_range"] = [1.0, 1.0]

    env = RL_Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        dt=env_cfg['dt'],
        substeps=env_cfg['substeps'],
        show_viewer=True,
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

def eval_policy_with_data_collection(env, policy, args):
    """reset直後のobsも含めて保存し、その後は各step後のobs/torqueを保存"""
    step_data = []
    obs_data = []
    torque_data = []

    # reset直後を記録（step=0）
    obs, _ = env.reset()
    cnt = 0

    step_data.append(cnt)
    obs_data.append(obs.cpu().numpy().flatten())
    torque_data.append(_read_torques(env))

    print(f"データ収集開始: {args.steps} ステップ")

    with torch.no_grad():
        for i in range(args.steps):
            # 行動計算 → 環境を1ステップ進める
            actions = policy(obs)
            #actions = actions * args.scale  # スケール調整
            actions = actions * 0.0  # スケール調整
            obs, rews, dones, infos = env.step(actions)

            # ステップ後のデータを記録
            cnt += 1
            step_data.append(cnt)
            obs_data.append(obs.cpu().numpy().flatten())
            torque_data.append(_read_torques(env))

            if i % 20 == 0:
                print(f"Step {i+1}/{args.steps}, Total steps: {cnt}")

            # 終了時はすぐresetし、reset直後のobsも記録して継続
            if dones.any():
                obs, _ = env.reset()
                cnt += 1
                step_data.append(cnt)
                obs_data.append(obs.cpu().numpy().flatten())
                torque_data.append(_read_torques(env))

    print(f"データ収集完了: {len(step_data)} rows")

    # CSVファイルに保存（トルク付き）
    df = save_simple_csv(step_data, obs_data, args.exp_name, args.ckpt, torque_data, args.scale)
    return df

def eval_policy_continuous(env, policy, args):
    """連続評価（データ収集なし）"""
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            actions = actions * args.scale  # スケール調整
            obs, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    env, policy, args = main()
    
    # データ収集付き評価を実行
    # df = eval_policy_with_data_collection(env, policy, args)
    
    # 必要に応じて連続評価も実行
    eval_policy_continuous(env, policy, args)

"""
# 使用例:
# データ収集付き評価(100ステップ)
python3 ishiki_eval_gs.py -e ishiki-walking-no-vel --ckpt 2000 --steps 100 --scale 0.5

# データ収集付き評価(500ステップ)
python3 ishiki_eval_gs.py -e ishiki-walking-no-vel --ckpt 2000 --steps 500 --scale 0.5
"""
