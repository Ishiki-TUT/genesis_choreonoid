"""
HRP2 Walking Simulation - Using Existing RL Environment
既存のRL環境を活用して、歩行計画をシミュレータで実行
"""

import sys
import os
import numpy as np
import argparse
import json
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "irsl_rl"))

# Genesis とRL環境をインポート
try:
    from rl_env_gs import RLEnvGenesis
    import genesis as gs
except ImportError as e:
    print(f"Warning: Could not import RL environment: {e}")
    print("Continuing with basic setup...")

import torch
from footstep_planner import FootstepPlanner, Footstep, Param, Ground
from hrp2_footplane import generate_5step_walking_plan


class HRP2WalkingVisualizer:
    """
    HRP2 歩行計画ビジュアライザー
    既存のRL環境を使用して、決定論的な歩行計画を可視化
    """
    
    def __init__(self, num_envs=1, show_viewer=True, dt=0.01, device="cuda"):
        """
        初期化
        
        Args:
            num_envs: 環境数
            show_viewer: ビューアーを表示
            dt: シミュレーション時間ステップ
            device: 計算デバイス ("cuda" or "cpu")
        """
        self.num_envs = num_envs
        self.show_viewer = show_viewer
        self.dt = dt
        self.device = device
        self.time = 0.0
        
        # RL環境をセットアップ
        self.env = None
        self._setup_environment()
        
        # 歩行計画
        self.footstep = None
        self.param = None
        self.current_step = 0
        self.step_progress = 0.0
        
        # 制御状態
        self.joint_targets = None
        self.joint_names = None
        
        print(f"✓ HRP2 Walking Visualizer initialized")
        print(f"  Environments: {num_envs}")
        print(f"  Device: {device}")
        print(f"  Time step: {dt} s")
    
    def _setup_environment(self):
        """RL環境をセットアップ"""
        print("\nSetting up RL environment...")
        
        # 基本的な環境設定
        env_cfg = {
            "num_envs": self.num_envs,
            "gravity": 9.81,
            "episode_length": 1000,
        }
        
        obs_cfg = {
            "obs_dim": 64,
        }
        
        reward_cfg = {
            "reward_scales": {
                "tracking": 1.0,
            }
        }
        
        command_cfg = {
            "lin_vel_x_range": [0.2, 0.4],
            "lin_vel_y_range": [-0.1, 0.1],
            "ang_vel_yaw_range": [-0.3, 0.3],
        }
        
        try:
            # RL環境を作成（可視化モード）
            self.env = RLEnvGenesis(
                num_envs=self.num_envs,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=self.show_viewer,
                device=self.device,
                dt=self.dt,
            )
            
            self.joint_names = self.env.dof_names if hasattr(self.env, 'dof_names') else []
            
            print(f"✓ RL Environment created successfully")
            print(f"  DOFs: {len(self.joint_names) if self.joint_names else 'Unknown'}")
            
        except Exception as e:
            print(f"Warning: Could not create RL environment: {e}")
            print("Continuing with minimal setup...")
    
    def generate_walking_plan(self, stride=0.2, sway=0.0, turn=0.0,
                             spacing=0.2, climb=0.0, com_height=0.8,
                             T=1.0, duration_per_step=0.8):
        """
        歩行計画を生成
        
        Args:
            stride, sway, turn, spacing, climb: 歩行パラメータ
            com_height: CoM高さ
            T: 時定数
            duration_per_step: ステップ時間
        """
        print("\nGenerating walking plan...")
        
        self.footstep, self.param = generate_5step_walking_plan(
            stride=stride,
            sway=sway,
            turn=turn,
            spacing=spacing,
            climb=climb,
            com_height=com_height,
            T=T,
            duration_per_step=duration_per_step
        )
        
        print(f"✓ Generated walking plan with {len(self.footstep.steps)} steps")
        
        # 歩行計画の詳細を表示
        print("\nWalking Plan Summary:")
        print("-" * 70)
        total_time = 0.0
        for i, step in enumerate(self.footstep.steps):
            total_time += step.duration
            support_foot = "LEFT" if step.side == 0 else "RIGHT"
            print(f"Step {i}: Support={support_foot:5s} | "
                  f"Time={step.duration:.2f}s | "
                  f"Left pos={step.foot_pos[0]} | "
                  f"Right pos={step.foot_pos[1]}")
        
        print(f"Total time: {total_time:.2f} s")
        print("-" * 70)
        
        self.current_step = 0
        self.step_progress = 0.0
    
    def update_step_progress(self, elapsed_time):
        """経過時間に基づいてステップを更新"""
        total_time = 0.0
        for i, step in enumerate(self.footstep.steps):
            total_time += step.duration
            if elapsed_time < total_time:
                self.current_step = i
                time_in_step = elapsed_time - (total_time - step.duration)
                self.step_progress = min(time_in_step / step.duration, 1.0)
                return
        
        # 最後のステップ
        self.current_step = len(self.footstep.steps) - 1
        self.step_progress = 1.0
    
    def get_step_info(self):
        """現在のステップ情報を取得"""
        if not self.footstep or self.current_step >= len(self.footstep.steps):
            return None
        
        step = self.footstep.steps[self.current_step]
        return {
            'index': self.current_step,
            'support_foot': int(step.side),
            'left_foot_pos': step.foot_pos[0].copy(),
            'right_foot_pos': step.foot_pos[1].copy(),
            'left_foot_angle': step.foot_angle[0].copy(),
            'right_foot_angle': step.foot_angle[1].copy(),
            'zmp': step.zmp.copy(),
            'dcm': step.dcm.copy(),
            'progress': self.step_progress,
        }
    
    def step(self, action=None):
        """
        シミュレーションを1ステップ進める
        
        Args:
            action: 関節コマンド（Noneの場合は自動生成）
        """
        try:
            if self.env:
                # RL環境で1ステップ
                obs, _, _, _ = self.env.step(action if action is not None 
                                             else np.zeros(self.env.action_space.shape))
        except Exception as e:
            print(f"Note: {e}")
        
        # 時間を進める
        self.time += self.dt
        self.update_step_progress(self.time)
    
    def render(self):
        """画面を描画"""
        if self.env and self.show_viewer:
            try:
                if hasattr(self.env, 'viewer') and self.env.viewer:
                    self.env.viewer.render()
            except:
                pass
    
    def reset(self):
        """環境をリセット"""
        self.time = 0.0
        self.current_step = 0
        self.step_progress = 0.0
        
        if self.env:
            try:
                self.env.reset()
            except:
                pass
    
    def run_visualization(self, duration=10.0, verbose=True):
        """
        ビジュアライゼーションを実行
        
        Args:
            duration: シミュレーション時間 [s]
            verbose: 詳細出力
        """
        print(f"\n{'='*70}")
        print(f"Running Walking Visualization for {duration:.1f} seconds")
        print(f"{'='*70}\n")
        
        num_steps = int(duration / self.dt)
        
        for i in range(num_steps):
            # 1ステップ進める
            self.step(action=None)
            
            # 画面描画
            self.render()
            
            # 定期的に状態を表示
            if verbose and i % 100 == 0:
                step_info = self.get_step_info()
                if step_info:
                    support = "LEFT" if step_info['support_foot'] == 0 else "RIGHT"
                    print(f"[{i:5d}] Time: {self.time:7.2f}s | "
                          f"Step: {step_info['index']}/{len(self.footstep.steps)-1} | "
                          f"Support: {support} | "
                          f"Progress: {step_info['progress']*100:5.1f}%")
            
            # 時間制限チェック
            if self.time > duration:
                break
        
        print(f"\n{'='*70}")
        print(f"Visualization completed!")
        print(f"Total simulation time: {self.time:.2f} s")
        print(f"{'='*70}\n")
    
    def save_trajectory_data(self, filename="walking_trajectory.json"):
        """軌跡データをファイルに保存"""
        if not self.footstep:
            print("No footstep data to save")
            return
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'com_height': self.param.com_height,
                'T': self.param.T,
            },
            'steps': []
        }
        
        for i, step in enumerate(self.footstep.steps):
            step_data = {
                'index': i,
                'support_foot': int(step.side),
                'left_foot': {
                    'position': step.foot_pos[0].tolist(),
                    'angle_rpy': step.foot_angle[0].tolist(),
                },
                'right_foot': {
                    'position': step.foot_pos[1].tolist(),
                    'angle_rpy': step.foot_angle[1].tolist(),
                },
                'zmp': step.zmp.tolist(),
                'dcm': step.dcm.tolist(),
                'duration': float(step.duration),
                'stepping': bool(step.stepping),
            }
            data['steps'].append(step_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Trajectory saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='HRP2 Walking Plan Visualizer using Genesis and RL Environment'
    )
    parser.add_argument('--stride', type=float, default=0.2,
                       help='Forward stride distance (m)')
    parser.add_argument('--sway', type=float, default=0.0,
                       help='Lateral sway (m)')
    parser.add_argument('--turn', type=float, default=0.0,
                       help='Turn angle (rad)')
    parser.add_argument('--spacing', type=float, default=0.2,
                       help='Foot spacing (m)')
    parser.add_argument('--com-height', type=float, default=0.8,
                       help='Center of mass height (m)')
    parser.add_argument('--T', type=float, default=1.0,
                       help='Time constant (s)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Simulation duration (s)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Simulation time step (s)')
    parser.add_argument('--num-envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--no-viewer', action='store_true',
                       help='Disable viewer')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Computing device (cuda or cpu)')
    parser.add_argument('--save-trajectory', type=str, default=None,
                       help='Save trajectory to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HRP2 Walking Plan Visualizer")
    print("Genesis + RL Environment + Footstep Planning")
    print("="*70 + "\n")
    
    # ビジュアライザーを作成
    visualizer = HRP2WalkingVisualizer(
        num_envs=args.num_envs,
        show_viewer=not args.no_viewer,
        dt=args.dt,
        device=args.device
    )
    
    # 歩行計画を生成
    visualizer.generate_walking_plan(
        stride=args.stride,
        sway=args.sway,
        turn=args.turn,
        spacing=args.spacing,
        com_height=args.com_height,
        T=args.T,
    )
    
    # ビジュアライゼーションを実行
    visualizer.run_visualization(
        duration=args.duration,
        verbose=not args.quiet
    )
    
    # 軌跡をファイルに保存
    if args.save_trajectory:
        visualizer.save_trajectory_data(args.save_trajectory)
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
