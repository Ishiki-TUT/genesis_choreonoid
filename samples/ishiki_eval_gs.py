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

def save_simple_csv(step_data, action_data, obs_data, exp_name, ckpt):
    """CSVファイルにデータを保存する関数"""
    if not step_data:
        print("データがありません")
        return None
    
    # 基本的な辞書形式でデータを整理
    data_dict = {'step': step_data}
    
    # Actionデータ
    action_array = np.array(action_data)
    for i in range(action_array.shape[1]):
        data_dict[f'action_{i}'] = action_array[:, i]
    
    # Observationデータ
    obs_array = np.array(obs_data)
    for i in range(obs_array.shape[1]):
        data_dict[f'obs_{i}'] = obs_array[:, i]
    
    # DataFrameを作成して保存
    df = pd.DataFrame(data_dict)
    
    # obs_dataディレクトリを作成
    os.makedirs('obs_data', exist_ok=True)
    
    csv_filename = f'obs_data/genesis_{exp_name}_ckpt{ckpt}_simple.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"データを保存しました: {csv_filename}")
    print(f"データ形状: {df.shape}")
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="kawada-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100, help="データ収集ステップ数")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

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

def eval_policy_with_data_collection(env, policy, args):
    """データ収集付きの評価関数"""
    # データ収集用のリスト
    step_data = []
    action_data = []
    obs_data = []
    
    obs, _ = env.reset()
    cnt = 0
    
    print(f"データ収集開始: {args.steps} ステップ")
    
    with torch.no_grad():
        for i in range(args.steps):
            # ポリシーから行動を取得
            actions = policy(obs)
            
            # データを記録
            step_data.append(cnt)
            action_data.append(actions.cpu().numpy().flatten())
            obs_data.append(obs.cpu().numpy().flatten())
            
            # 環境をステップ実行
            obs, rews, dones, infos = env.step(actions)
            
            # 進行状況を表示
            if i % 20 == 0:
                print(f"Step {i+1}/{args.steps}, Total steps: {cnt}")
            
            cnt += 1
            
            # リセット条件をチェック
            if dones.any():
                print(f"Episode finished at step {cnt}, resetting...")
                obs, _ = env.reset()
    
    print(f"データ収集完了: {len(step_data)} steps collected")
    
    # CSVファイルに保存
    df = save_simple_csv(step_data, action_data, obs_data, args.exp_name, args.ckpt)
    
    return df

def eval_policy_continuous(env, policy):
    """連続評価（データ収集なし）"""
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    env, policy, args = main()
    
    # データ収集付き評価を実行
    df = eval_policy_with_data_collection(env, policy, args)
    
    # 必要に応じて連続評価も実行
    # eval_policy_continuous(env, policy)

"""
# 使用例:
# データ収集付き評価(100ステップ)
python3 ishiki_eval_gs.py -e ishiki-walking-no-vel --ckpt 2000 --steps 100

# データ収集付き評価(500ステップ)
python3 ishiki_eval_gs.py -e ishiki-walking-no-vel --ckpt 2000 --steps 500
"""
