"""
HRP2 Deterministic Walking Trajectory Generator
C++ の footstep_planner.cpp と stepping_controller.cpp をポートした
決定論的な歩行軌道生成
"""

import argparse
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from datetime import datetime

from footstep_planner import (
    FootstepPlanner, Step, Footstep, Param, Ground, 
    rotate_vector, rpy_to_quat, quat_to_rpy
)
from stepping_controller import (
    SteppingController, Timer, Centroid, Base, Foot
)


def generate_5step_walking_plan(
    stride=0.05, sway=0.0, turn=0.0, 
    spacing=0.2, com_height=0.65, T=1.0, 
    duration_per_step=0.8
):
    """
    5ステップの歩行計画を生成
    
    Args:
        stride: 前進距離 [m]
        sway: 左右のスウェイ幅 [m]
        turn: 旋回角度 [rad]
        spacing: 足間隔 [m]
        com_height: CoM高さ [m]
        T: 時間定数 [s]
        duration_per_step: 1ステップの時間 [s]
    
    Returns:
        footstep: Footstepオブジェクト（複数のStepを含む）
        param: Paramオブジェクト
    """
    
    param = Param(com_height=com_height, T=T)
    steps = []
    
    # 初期位置
    current_x = 0.0
    current_y = 0.0
    current_time = 0.0
    
    print("\n" + "="*70)
    print("Debug: Generating 5-step walking plan")
    print(f"  stride={stride}, spacing={spacing}, duration_per_step={duration_per_step}")
    print("="*70)
    
    # ★重要: ステップを逐次的に生成（5ステップ）
    for step_idx in range(5):
        step = Step()
        
        # 【重要】各ステップの位置を計算
        if step_idx == 0:
            # ステップ0（初期左足）
            current_x = 0.0
            current_y = spacing / 2  # 左足は+Y
            step.stepping = False
            
        elif step_idx == 1:
            # ステップ1（右足）- 前進
            current_x += stride
            current_y = -spacing / 2  # 右足は-Y
            step.stepping = True
            
        elif step_idx == 2:
            # ステップ2（左足）- 前進
            current_x += stride
            current_y = spacing / 2  # 左足は+Y
            step.stepping = True
            
        elif step_idx == 3:
            # ステップ3（右足）- 前進
            current_x += stride
            current_y = -spacing / 2  # 右足は-Y
            step.stepping = True
            
        elif step_idx == 4:
            # ステップ4（左足）- 前進
            current_x += stride
            current_y = spacing / 2  # 左足は+Y
            step.stepping = True
        
        # ZMP位置を設定
        step.zmp = np.array([current_x, current_y, 0.0], dtype=np.float32)
        
        # DCM位置を初期値として設定
        step.dcm = step.zmp.copy()
        
        # 時間を設定
        step.time = current_time
        current_time += duration_per_step
        
        steps.append(step)
        
        # デバッグ出力
        foot_name = "LEFT" if step_idx % 2 == 0 else "RIGHT"
        print(f"Step {step_idx} ({foot_name}): ZMP=[{step.zmp[0]:.4f}, {step.zmp[1]:.4f}], time={step.time:.2f}s, stepping={step.stepping}")
    
    print(f"Total steps generated: {len(steps)}")
    print("="*70 + "\n")
    
    footstep = Footstep(steps=steps)
    
    return footstep, param


def format_step_info(step: Step, step_idx: int) -> str:
    """ステップ情報をフォーマット"""
    left_pos = step.foot_pos[0]
    right_pos = step.foot_pos[1]
    left_angle = step.foot_angle[0]
    right_angle = step.foot_angle[1]
    
    info = f"""
Step {step_idx}:
  Support foot: {'LEFT' if step.side == 0 else 'RIGHT'}
  
  Left foot:
    Position: [{left_pos[0]:7.4f}, {left_pos[1]:7.4f}, {left_pos[2]:7.4f}]
    Angle (RPY): [{left_angle[0]:7.4f}, {left_angle[1]:7.4f}, {left_angle[2]:7.4f}]
  
  Right foot:
    Position: [{right_pos[0]:7.4f}, {right_pos[1]:7.4f}, {right_pos[2]:7.4f}]
    Angle (RPY): [{right_angle[0]:7.4f}, {right_angle[1]:7.4f}, {right_angle[2]:7.4f}]
  
  ZMP:   [{step.zmp[0]:7.4f}, {step.zmp[1]:7.4f}, {step.zmp[2]:7.4f}]
  DCM:   [{step.dcm[0]:7.4f}, {step.dcm[1]:7.4f}, {step.dcm[2]:7.4f}]
  
  Duration: {step.duration:.4f} s
  Stepping: {step.stepping}
"""
    return info


def save_trajectory_to_file(footstep: Footstep, param: Param, output_file: str):
    """軌跡を JSON ファイルに保存"""
    trajectory_data = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'com_height': param.com_height,
            'T': param.T,
        },
        'steps': []
    }
    
    for i, step in enumerate(footstep.steps):
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
        trajectory_data['steps'].append(step_data)
    
    with open(output_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f"Trajectory saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='HRP2 Walking Simulation')
    parser.add_argument('--stride', type=float, default=0.1, help='Stride distance (m)')  # ← デフォルト0.1に変更
    parser.add_argument('--sway', type=float, default=0.0, help='Sway (m)')
    parser.add_argument('--turn', type=float, default=0.0, help='Turn angle (rad)')
    parser.add_argument('--spacing', type=float, default=0.2, help='Foot spacing (m)')
    parser.add_argument('--com-height', type=float, default=0.65, help='CoM height (m)')
    parser.add_argument('--T', type=float, default=0.4, help='Time constant (s)')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration (s)')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step (s)')
    parser.add_argument('--no-viewer', action='store_true', help='Disable viewer')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HRP2 Walking Simulation with Footstep Planning")
    print("="*70)
    
    sim = HRP2WalkingSimulator(dt=args.dt, render=not args.no_viewer)
    
    sim.generate_walking_plan(
        stride=args.stride,  # ← コマンドライン引数から取得
        sway=args.sway,
        turn=args.turn,
        spacing=args.spacing,
        com_height=args.com_height,
        T=args.T,
        duration_per_step=0.8
    )
    
    sim.run(duration=args.duration)
    print("="*70)
    


def generate_walking_plan(self, stride=0.05, sway=0.0, turn=0.0, 
                          spacing=0.2, com_height=0.65, T=1.0, 
                          duration_per_step=0.8):
    """歩行計画を生成"""
    print("\n" + "="*70)
    print("Generating walking plan...")
    print("="*70)
    
    # ★注意: strideを大きめに設定（デフォルト0.05は小さすぎる）
    self.footstep, self.param = generate_5step_walking_plan(
        stride=stride,  # ← ここで指定される値を使用
        sway=sway,
        turn=turn,
        spacing=spacing,
        com_height=com_height,
        T=T,
        duration_per_step=duration_per_step
    )
    
    # 【デバッグ】生成されたステップを確認
    print(f"\n✓ Generated {len(self.footstep.steps)} steps")
    print("\nFootstep Details:")
    print("-" * 70)
    for i, step in enumerate(self.footstep.steps):
        print(f"Step {i:2d}: ZMP=[{step.zmp[0]:.4f}, {step.zmp[1]:.4f}, {step.zmp[2]:.4f}]  " +
              f"DCM=[{step.dcm[0]:.4f}, {step.dcm[1]:.4f}, {step.dcm[2]:.4f}]  " +
              f"Stepping={getattr(step, 'stepping', 'N/A')}")
    print("-" * 70)
    
    # ステップの設定
    for i, step in enumerate(self.footstep.steps):
        if i == 0:
            step.stepping = False
        else:
            step.stepping = True
    
    # DCMとZMP軌道を生成
    planner = FootstepPlanner()
    planner.generate_dcm(self.param, self.footstep)
    
    print(f"✓ Parameters:")
    print(f"  CoM Height: {self.param.com_height:.4f} m")
    print(f"  Time Constant T: {self.param.T:.4f} s")
    
    # コントローラの初期化
    self._setup_stepping_controller()
    
    # 最初のステップの状態を反映
    if len(self.footstep.steps) > 0:
        st0 = self.footstep.steps[0]
        self.centroid.zmp_ref = st0.zmp.copy()
        self.centroid.dcm_ref = st0.dcm.copy()
        self.centroid.zmp_target = st0.zmp.copy()
        self.centroid.dcm_target = st0.dcm.copy()
        
        self.com_pos = np.array([st0.dcm[0], st0.dcm[1], com_height])
    
    self.current_step = 0
    self.step_progress = 0.0
    print("="*70 + "\n")
    


if __name__ == "__main__":
    footstep, param = main()