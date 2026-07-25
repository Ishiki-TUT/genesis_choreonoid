#!/usr/bin/env python3
"""
HRP2 Walking Plan Simple Visualizer
Genesis上で歩行計画を実行する最もシンプルなバージョン
"""

import sys
import os
import numpy as np
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "irsl_rl"))

import genesis as gs
import torch

from footstep_planner import FootstepPlanner, Footstep, Param, Ground
from hrp2_footplane import generate_5step_walking_plan


class SimpleWalkingVisualizer:
    """シンプルな歩行計画ビジュアライザー"""
    
    def __init__(self, urdf_path, dt=0.01, render=True):
        """
        初期化
        
        Args:
            urdf_path: ロボットURDFファイルのパス
            dt: シミュレーション時間ステップ [s]
            render: ビューアーを表示するか
        """
        self.dt = dt
        self.render = render
        self.time = 0.0
        self.scene = None
        self.robot = None
        
        # 歩行計画
        self.footstep = None
        self.param = None
        self.current_step = 0
        self.step_progress = 0.0
        
        print("Initializing Genesis...")
        gs.init(backend=gs.cuda)
        
        # Scene を作成
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt),
            show_viewer=self.render,
        )
        
        # 地面を追加
        self.ground = self.scene.add_entity(gs.morphs.Plane())
        
        # ロボットを追加
        print(f"Loading URDF: {urdf_path}")
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, fixed=False)
        )
        
        # Scene を構築
        self.scene.build()
        
        # PD制御パラメータを設定
        self._setup_pd_control()
        
        # 初期姿勢を設定（床にめり込まないように）
        self._set_initial_pose()
        
        print("✓ Genesis initialized successfully")
    
    def _setup_pd_control(self):
        """PD制御パラメータを設定"""
        try:
            if not self.robot:
                return
            
            # HRP2用の PD制御ゲイン
            # 脚関節：Kp=500, Kv=20
            # 体幹関節：Kp=300, Kv=15
            
            n_dofs = len(self.robot.dof_names())
            
            # すべての関節に対して PD ゲインを設定
            kp = torch.full((n_dofs,), 300.0, dtype=torch.float32, device='cuda')
            kv = torch.full((n_dofs,), 15.0, dtype=torch.float32, device='cuda')
            
            # 脚関節（通常は後ろの方のDOF）はゲインを上げる
            leg_kp = 500.0
            leg_kv = 20.0
            
            for i, dof_name in enumerate(self.robot.dof_names()):
                if 'LEG' in dof_name or 'leg' in dof_name:
                    kp[i] = leg_kp
                    kv[i] = leg_kv
            
            # PD制御を有効化
            self.robot.set_dofs_kp(kp.cpu().numpy())
            self.robot.set_dofs_kv(kv.cpu().numpy())
            
            print(f"✓ PD control configured: Kp={kp[0]:.1f}, Kv={kv[0]:.1f}")
        except Exception as e:
            print(f"Note: Could not setup PD control: {e}")
    
    def _set_initial_pose(self):
        """初期姿勢を設定（床にめり込まないように）"""
        try:
            if not self.robot:
                return
            
            n_dofs = len(self.robot.dof_names())
            
            # デフォルト姿勢：立ち姿勢
            # 股関節ピッチを少し上げて、膝を曲げる
            initial_angles = np.zeros(n_dofs)
            
            # HRP2の脚関節を立ち姿勢に設定
            for i, dof_name in enumerate(self.robot.dof_names()):
                if 'PITCH' in dof_name and 'KNEE' in dof_name:
                    # 膝を少し曲げる（0.3 rad）
                    initial_angles[i] = 0.3
                elif 'PITCH' in dof_name and 'HIP' in dof_name:
                    # 股関節ピッチを少し下げる（-0.15 rad）
                    initial_angles[i] = -0.15
            
            # 関節角度を設定
            target_angles = torch.from_numpy(initial_angles).float().to('cuda')
            for i, angle in enumerate(initial_angles):
                try:
                    self.robot.set_dof_target(torch.tensor([angle], dtype=torch.float32, device='cuda'), i)
                except:
                    pass
            
            # シミュレーションを数ステップ実行して落ち着かせる
            print("Settling initial pose...")
            for _ in range(100):
                self.scene.step()
            
            print("✓ Initial pose set successfully")
        except Exception as e:
            print(f"Note: Could not set initial pose: {e}")
    
    def generate_walking_plan(self, stride=0.2, spacing=0.2, com_height=0.8):
        """歩行計画を生成"""
        print("\nGenerating walking plan...")
        self.footstep, self.param = generate_5step_walking_plan(
            stride=stride,
            spacing=spacing,
            com_height=com_height
        )
        print(f"✓ Generated {len(self.footstep.steps)} steps")
        
        self.current_step = 0
        self.step_progress = 0.0
    
    def update_step(self, elapsed_time):
        """経過時間に基づいてステップを更新"""
        total_time = 0.0
        for i, step in enumerate(self.footstep.steps):
            total_time += step.duration
            if elapsed_time < total_time:
                self.current_step = i
                time_in_step = elapsed_time - (total_time - step.duration)
                self.step_progress = min(time_in_step / step.duration, 1.0)
                return
        
        self.current_step = len(self.footstep.steps) - 1
        self.step_progress = 1.0
    
    def step(self):
        """シミュレーションを1ステップ進める"""
        self.update_step(self.time)
        
        # 関節目標を更新
        self._update_joint_targets()
        
        self.scene.step()
        self.time += self.dt
    
    def _update_joint_targets(self):
        """現在のステップに基づいて関節目標を更新"""
        try:
            if not self.robot or not self.footstep:
                return
            
            step = self.footstep.steps[self.current_step]
            n_dofs = len(self.robot.dof_names())
            
            # 目標角度を初期化
            target_angles = np.zeros(n_dofs)
            
            # スウィング足の持ち上げ動作を制御
            if step.stepping:
                swg = 1 - step.side
                
                # スウィング足の股関節ピッチを制御
                hip_pitch_swing = -0.4 if self.step_progress < 0.5 else -0.2
                
                for i, dof_name in enumerate(self.robot.dof_names()):
                    if swg == 0 and 'LLEG' in dof_name and 'PITCH' in dof_name and 'HIP' in dof_name:
                        # 左足のヒップピッチ
                        target_angles[i] = hip_pitch_swing
                    elif swg == 1 and 'RLEG' in dof_name and 'PITCH' in dof_name and 'HIP' in dof_name:
                        # 右足のヒップピッチ
                        target_angles[i] = hip_pitch_swing
                    elif 'KNEE' in dof_name and 'PITCH' in dof_name:
                        # 膝ピッチを維持
                        target_angles[i] = 0.3
            
            # 目標角度を関節に適用
            for i, angle in enumerate(target_angles):
                try:
                    self.robot.set_dof_target(
                        torch.tensor([angle], dtype=torch.float32, device='cuda'),
                        i
                    )
                except:
                    pass
        except Exception as e:
            pass  # エラーを無視
    
    def render_frame(self):
        """画面を描画"""
        try:
            if self.render and hasattr(self.scene, 'viewer') and self.scene.viewer:
                self.scene.viewer.render()
        except:
            pass
    
    def run(self, duration=10.0, verbose=True):
        """シミュレーションを実行"""
        print(f"\nRunning simulation for {duration} seconds...\n")
        
        num_steps = int(duration / self.dt)
        
        for i in range(num_steps):
            self.step()
            self.render_frame()
            
            if verbose and i % 100 == 0:
                step_info = self.footstep.steps[self.current_step]
                support = "LEFT" if step_info.side == 0 else "RIGHT"
                print(f"[{i:5d}] Time: {self.time:7.2f}s | "
                      f"Step: {self.current_step}/{len(self.footstep.steps)-1} | "
                      f"Support: {support} | "
                      f"Progress: {self.step_progress*100:5.1f}%")
            
            if self.time > duration:
                break
        
        print(f"\n✓ Simulation completed!")
        print(f"Total time: {self.time:.2f} s")


def main():
    parser = argparse.ArgumentParser(
        description='HRP2 Walking Plan Visualizer (Simple Version)'
    )
    parser.add_argument('--urdf', type=str, 
                       default='hrp2_description/HRP2_genesis.urdf',
                       help='Path to URDF file')
    parser.add_argument('--stride', type=float, default=0.2,
                       help='Stride distance (m)')
    parser.add_argument('--spacing', type=float, default=0.2,
                       help='Foot spacing (m)')
    parser.add_argument('--com-height', type=float, default=0.8,
                       help='CoM height (m)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Simulation duration (s)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step (s)')
    parser.add_argument('--no-viewer', action='store_true',
                       help='Disable viewer')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HRP2 Walking Plan Visualizer")
    print("="*70)
    
    # フルパスを構築
    urdf_path = os.path.join(os.path.dirname(__file__), args.urdf)
    
    # ビジュアライザーを作成
    visualizer = SimpleWalkingVisualizer(
        urdf_path=urdf_path,
        dt=args.dt,
        render=not args.no_viewer
    )
    
    # 歩行計画を生成
    visualizer.generate_walking_plan(
        stride=args.stride,
        spacing=args.spacing,
        com_height=args.com_height
    )
    
    # シミュレーションを実行
    visualizer.run(
        duration=args.duration,
        verbose=not args.quiet
    )
    
    print("="*70)
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
