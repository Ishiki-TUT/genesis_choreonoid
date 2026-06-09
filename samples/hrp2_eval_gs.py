import argparse
import os
import pickle

from importlib import metadata
ver_rsl_rl = metadata.version("rsl-rl-lib")
print(f"Version of rsl-rl-lib : {ver_rsl_rl}")

import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

# ★変更: HRP2用のEnv定義をインポート
from hrp2_env_gs import BP000EnvGenesis as RLEnv

def main():
    parser = argparse.ArgumentParser()
    # ★変更: デフォルトの実験名を変更
    parser.add_argument("-e", "--exp_name", type=str, default="hrp2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--action_scale", type=float, default=1.0)
    args = parser.parse_args()

    ## set robot path fix collisiton 
    # ★変更: HRP2のURDFパスを指定
    robot_path = os.path.join(os.path.dirname(__file__), "hrp2_description/HRP2_genesis.urdf")

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    
    # ログディレクトリの存在確認
    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        return None, None, args
        
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    global action_scale
    action_scale = args.action_scale

    ## override
    env_cfg["episode_length_s"] = 40.0
    command_cfg["lin_vel_x_range"] = [0.5, 0.5]
    env_cfg['base_roll_noise'] = [0,0]
    env_cfg['base_pitch_noise'] = [0,0]

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
    
    if not os.path.exists(resume_path):
        print(f"Error: Checkpoint not found: {resume_path}")
        return env, None, args

    runner.load(resume_path)
    policy = runner.get_inference_policy(device='cuda')

    return env, policy, args

def eval_policy(env, policy, num_steps=500):
    """
    ポリシーを評価し、歩行メトリクスを記録
    """
    if policy is None:
        return None
    
    # メトリクス記録用の辞書
    metrics = {
        'lin_vel': [],
        'command_vel': [],
        'base_height': [],
        'base_roll': [],
        'base_pitch': [],
        'base_yaw': [],
        'left_foot_z': [],
        'right_foot_z': [],
        'action_magnitude': [],
        'rewards': [],
    }
    
    obs, _ = env.reset()
    
    print("Starting policy evaluation...")
    print(f"Running {num_steps} steps...\n")
    
    with torch.no_grad():
        for cnt in range(num_steps):
            # アクション取得
            actions = policy(obs)
            scaled_actions = actions * action_scale
            
            # ステップ実行
            obs, rews, dones, infos = env.step(scaled_actions)
            
            # メトリクスを記録
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", obs)
            else:
                obs_tensor = obs
            
            # 環境から状態情報を取得
            try:
                metrics['lin_vel'].append(env.base_lin_vel[:, 0].cpu().float())  # X方向速度
                metrics['command_vel'].append(env.commands[:, 0].cpu().float())   # 指令速度
                metrics['base_height'].append(env.base_pos[:, 2].cpu().float())   # ベース高さ
                metrics['base_roll'].append(env.projected_gravity[:, 0].cpu().float())
                metrics['base_pitch'].append(env.projected_gravity[:, 1].cpu().float())
                metrics['action_magnitude'].append(torch.norm(scaled_actions, dim=1).cpu().float())
                metrics['rewards'].append(rews.cpu().float())
                
                # 足の高さを記得（滞空時間判定用）
                if hasattr(env, 'l_ankle_z') and hasattr(env, 'r_ankle_z'):
                    metrics['left_foot_z'].append(env.l_ankle_z.cpu().float())
                    metrics['right_foot_z'].append(env.r_ankle_z.cpu().float())
                else:
                    print(f"Warning: Ankle height not available in environment")
            except Exception as e:
                print(f"Warning: Could not record metrics at step {cnt}: {e}")
            
            if (cnt + 1) % 100 == 0:
                print(f"Step: {cnt + 1}/{num_steps}")
            
            if dones.any():
                print(f"Episode ended at step {cnt}")
                break
    
    # メトリクスをテンソル化
    for key in metrics:
        if metrics[key]:
            metrics[key] = torch.cat(metrics[key], dim=0)
    
    return metrics

if __name__ == "__main__":
    res = main()
    if res is not None and len(res) == 3:
        env, policy, args = res
        if policy is not None:
            metrics = eval_policy(env, policy, num_steps=500)
            
            # メトリクスを保存
            if metrics is not None:
                log_dir = f"logs/{args.exp_name}"
                os.makedirs(log_dir, exist_ok=True)
                
                metrics_file = os.path.join(log_dir, "eval_metrics.pkl")
                with open(metrics_file, 'wb') as f:
                    pickle.dump(metrics, f)
                print(f"\nMetrics saved to {metrics_file}")
                
                # 簡単な統計情報を表示
                print("\n" + "="*60)
                print("Evaluation Results Summary")
                print("="*60)
                if 'lin_vel' in metrics and len(metrics['lin_vel']) > 0:
                    vel_tensor = torch.cat(metrics['lin_vel']) if isinstance(metrics['lin_vel'][0], torch.Tensor) else torch.tensor(metrics['lin_vel'])
                    print(f"Average velocity: {torch.mean(vel_tensor):.4f} m/s")
                    print(f"Velocity std: {torch.std(vel_tensor):.4f} m/s")
                if 'left_foot_z' in metrics and len(metrics['left_foot_z']) > 0:
                    left_z_tensor = torch.cat(metrics['left_foot_z']) if isinstance(metrics['left_foot_z'][0], torch.Tensor) else torch.tensor(metrics['left_foot_z'])
                    right_z_tensor = torch.cat(metrics['right_foot_z']) if isinstance(metrics['right_foot_z'][0], torch.Tensor) else torch.tensor(metrics['right_foot_z'])
                    print(f"Left foot avg height: {torch.mean(left_z_tensor):.4f} m")
                    print(f"Right foot avg height: {torch.mean(right_z_tensor):.4f} m")
                print("="*60)
        else:
            print("Failed to load policy")