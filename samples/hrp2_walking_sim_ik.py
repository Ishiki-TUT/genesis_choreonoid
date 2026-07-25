"""
HRP2 Walking Simulation - Advanced Version with Inverse Kinematics
Genesis上で歩行計画に従うロボット制御（IK利用）
"""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "irsl_rl"))

import genesis as gs
import torch

from footstep_planner import FootstepPlanner, Step, Footstep, Param, Ground
from hrp2_footplane import generate_5step_walking_plan


class SimpleIKSolver:
    """簡易的な逆運動学ソルバー"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def forward_kinematics_leg(joint_angles):
        """
        HRP2の脚の順運動学（簡略版）
        
        Args:
            joint_angles: [hip_roll, hip_pitch, hip_yaw, knee_pitch, ankle_pitch, ankle_roll]
        
        Returns:
            end_effector_pos: [x, y, z]
        """
        # 簡略版：リンク長さ（これらは実際のHRP2値に合わせる必要がある）
        L_hip = 0.1  # 股から膝までの距離
        L_knee = 0.3  # 膝から足首までの距離
        L_ankle = 0.1  # 足首から足先までの距離
        
        # 簡略化：主にpitch軸の動きを考慮
        theta_hip = joint_angles[1]
        theta_knee = joint_angles[3]
        theta_ankle = joint_angles[4]
        
        # Y座標（hip_roll）
        y = L_hip * np.sin(joint_angles[0])
        
        # Z座標（gravity方向の高さ）
        z = -(L_hip * np.cos(theta_hip) + 
              L_knee * np.cos(theta_hip + theta_knee) + 
              L_ankle * np.cos(theta_hip + theta_knee + theta_ankle))
        
        # X座標（forward方向）
        x = (L_hip * np.sin(theta_hip) + 
             L_knee * np.sin(theta_hip + theta_knee) + 
             L_ankle * np.sin(theta_hip + theta_knee + theta_ankle))
        
        return np.array([x, y, z])
    
    @staticmethod
    def jacobian_leg(joint_angles):
        """
        脚のヤコビアン（数値微分版）
        """
        delta = 1e-6
        J = np.zeros((3, 6))
        
        for i in range(6):
            angles_plus = joint_angles.copy()
            angles_plus[i] += delta
            
            angles_minus = joint_angles.copy()
            angles_minus[i] -= delta
            
            pos_plus = SimpleIKSolver.forward_kinematics_leg(angles_plus)
            pos_minus = SimpleIKSolver.forward_kinematics_leg(angles_minus)
            
            J[:, i] = (pos_plus - pos_minus) / (2 * delta)
        
        return J
    
    @staticmethod
    def solve_ik(target_pos, initial_angles, max_iterations=100, learning_rate=0.01):
        """
        簡易的な逆運動学を解く（勾配法）
        
        Args:
            target_pos: 目標位置 [3]
            initial_angles: 初期関節角 [6]
            max_iterations: 最大反復回数
            learning_rate: 学習率
        
        Returns:
            joint_angles: 解いた関節角 [6]
        """
        angles = initial_angles.copy().astype(float)
        
        for iteration in range(max_iterations):
            # 順運動学
            current_pos = SimpleIKSolver.forward_kinematics_leg(angles)
            
            # 誤差
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < 1e-4:
                # 収束
                break
            
            # ヤコビアン
            J = SimpleIKSolver.jacobian_leg(angles)
            
            # 疑似逆行列
            try:
                J_pinv = np.linalg.pinv(J, rcond=1e-3)
            except:
                break
            
            # 関節角を更新
            d_angles = learning_rate * J_pinv @ error
            angles += d_angles
            
            # 関節角を制約
            angles = np.clip(angles, -np.pi, np.pi)
        
        return angles


class HRP2AdvancedWalkingSimulator:
    """HRP2 高度な歩行シミュレータ"""
    
    def __init__(self, dt=0.01, render=True, use_ik=True):
        """
        初期化
        
        Args:
            dt: シミュレーション時間ステップ
            render: ビューアーを表示
            use_ik: 逆運動学を使用
        """
        self.dt = dt
        self.render = render
        self.use_ik = use_ik
        self.scene = None
        self.robot = None
        self.viewer = None
        self.time = 0.0
        
        # ロボット情報
        self.dof_names = None
        self.joint_indices = {}
        self.n_dofs = 0
        
        # 歩行計画
        self.footstep = None
        self.param = None
        self.current_step = 0
        self.step_progress = 0.0
        
        # 制御状態
        self.joint_angles = None
        self.target_angles = None
        self.ik_solver = SimpleIKSolver() if use_ik else None
        
        self._init_genesis()
    
    def _init_genesis(self):
        """Genesis を初期化"""
        print("Initializing Genesis with Physics Engine...")
        gs.init(backend=gs.cuda)
        
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 3.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ) if self.render else None,
            sim_options=gs.options.SimOptions(
                dt=self.dt,
            ),
            show_viewer=self.render,
        )
        
        # 地面
        self.ground = self.scene.add_entity(gs.morphs.Plane())
        
        # HRP2 ロボット
        robot_path = os.path.join(os.path.dirname(__file__), "hrp2_description/HRP2_genesis.urdf")
        print(f"Loading URDF from: {robot_path}")
        
        try:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(file=robot_path, fixed=False)
            )
            print(f"✓ Successfully loaded HRP2 robot")
        except Exception as e:
            print(f"✗ Error loading URDF: {e}")
            raise
        
        self.scene.build()
        self._setup_robot_info()
        
        print(f"✓ Genesis initialized with {self.n_dofs} DOFs")
        
        if self.render:
            self.viewer = self.scene.viewer
    
    def _setup_robot_info(self):
        """ロボット情報をセットアップ"""
        self.dof_names = self.robot.dof_names()
        self.n_dofs = len(self.dof_names)
        
        # 足関節マッピング
        leg_joint_names = [
            "LLEG_JOINT0", "LLEG_JOINT1", "LLEG_JOINT2", 
            "LLEG_JOINT3", "LLEG_JOINT4", "LLEG_JOINT5",
            "RLEG_JOINT0", "RLEG_JOINT1", "RLEG_JOINT2",
            "RLEG_JOINT3", "RLEG_JOINT4", "RLEG_JOINT5",
        ]
        
        for leg_name in leg_joint_names:
            for i, dof_name in enumerate(self.dof_names):
                if leg_name in dof_name:
                    self.joint_indices[leg_name] = i
                    break
        
        print(f"Mapped {len(self.joint_indices)} leg joints")
    
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
        self.joint_angles = np.zeros(self.n_dofs)
        self.target_angles = np.zeros(self.n_dofs)
    
    def update_step(self, elapsed_time):
        """経過時間からステップを更新"""
        total_time = 0.0
        for i, step in enumerate(self.footstep.steps):
            total_time += step.duration
            if elapsed_time < total_time:
                self.current_step = min(i, len(self.footstep.steps) - 1)
                time_in_step = elapsed_time - (total_time - step.duration)
                self.step_progress = min(time_in_step / step.duration, 1.0)
                return
        
        self.current_step = len(self.footstep.steps) - 1
        self.step_progress = 1.0
    
    def compute_target_angles(self):
        """目標関節角を計算"""
        step = self.footstep.steps[self.current_step]
        
        # 左脚と右脚の脚関節インデックス
        left_leg_indices = [self.joint_indices.get(f"LLEG_JOINT{i}") 
                           for i in range(6) if f"LLEG_JOINT{i}" in self.joint_indices]
        right_leg_indices = [self.joint_indices.get(f"RLEG_JOINT{i}") 
                            for i in range(6) if f"RLEG_JOINT{i}" in self.joint_indices]
        
        # 足の目標位置
        left_pos_target = step.foot_pos[0]
        right_pos_target = step.foot_pos[1]
        
        # IKで関節角を計算
        if self.use_ik and self.ik_solver:
            try:
                if left_leg_indices:
                    current_left_angles = self.target_angles[left_leg_indices]
                    left_angles = self.ik_solver.solve_ik(left_pos_target, current_left_angles)
                    for idx, angle in zip(left_leg_indices, left_angles):
                        if idx is not None:
                            self.target_angles[idx] = angle
                
                if right_leg_indices:
                    current_right_angles = self.target_angles[right_leg_indices]
                    right_angles = self.ik_solver.solve_ik(right_pos_target, current_right_angles)
                    for idx, angle in zip(right_leg_indices, right_angles):
                        if idx is not None:
                            self.target_angles[idx] = angle
            except Exception as e:
                print(f"IK error: {e}")
    
    def step_simulation(self):
        """シミュレーションを1ステップ進める"""
        self.update_step(self.time)
        self.compute_target_angles()
        
        # 関節角を適用（PD制御的に）
        # ここでは簡略化
        if self.robot:
            try:
                # 目標角度を設定
                for joint_name, idx in self.joint_indices.items():
                    if idx < self.n_dofs:
                        # 簡略版：直接角度を設定
                        self.robot.set_dof_target(
                            torch.tensor([self.target_angles[idx]], dtype=torch.float32),
                            idx
                        )
            except Exception as e:
                pass  # エラーを無視
        
        # シミュレーション実行
        self.scene.step()
        self.time += self.dt
    
    def render_step(self):
        """画面を描画"""
        if self.render and self.viewer:
            self.viewer.render()
    
    def run(self, duration=10.0):
        """シミュレーションを実行"""
        print(f"\nRunning Advanced Walking Simulation for {duration} seconds...")
        
        num_steps = int(duration / self.dt)
        
        for i in range(num_steps):
            self.step_simulation()
            self.render_step()
            
            if i % 100 == 0:
                print(f"[{i:5d}] Time: {self.time:7.2f}s | "
                      f"Step: {self.current_step}/{len(self.footstep.steps)-1} | "
                      f"Progress: {self.step_progress:.2f}")
            
            if self.time > duration:
                break
        
        print(f"\n✓ Simulation completed!")
        print(f"Total simulation time: {self.time:.2f} s")


def main():
    parser = argparse.ArgumentParser(description='HRP2 Advanced Walking Simulation')
    parser.add_argument('--stride', type=float, default=0.2, help='Stride distance (m)')
    parser.add_argument('--spacing', type=float, default=0.2, help='Foot spacing (m)')
    parser.add_argument('--com-height', type=float, default=0.8, help='CoM height (m)')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration (s)')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step (s)')
    parser.add_argument('--no-viewer', action='store_true', help='Disable viewer')
    parser.add_argument('--no-ik', action='store_true', help='Disable IK')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HRP2 Advanced Walking Simulation with IK")
    print("="*70)
    
    # シミュレータを作成
    sim = HRP2AdvancedWalkingSimulator(
        dt=args.dt,
        render=not args.no_viewer,
        use_ik=not args.no_ik
    )
    
    # 歩行計画を生成
    sim.generate_walking_plan(
        stride=args.stride,
        spacing=args.spacing,
        com_height=args.com_height
    )
    
    # シミュレーションを実行
    sim.run(duration=args.duration)
    
    print("="*70)


if __name__ == "__main__":
    main()
