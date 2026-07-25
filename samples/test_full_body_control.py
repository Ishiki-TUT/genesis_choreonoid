#!/usr/bin/env python3
"""
Test Full Body Control - HRP2の全身制御テスト

このスクリプトで以下を確認：
1. 全30個の関節が正しく制御されているか
2. 腕・頭・腰がPD制御でしっかりダンピングされているか
3. ブラブラ揺れが抑制されているか
"""

import sys
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "irsl_rl"))

import genesis as gs
import torch


def test_full_body_control():
    """全身制御テスト"""
    print("="*70)
    print("HRP2 Full Body Control Test")
    print("="*70)
    
    # Genesis を初期化
    print("\n[1/5] Initializing Genesis...")
    gs.init(backend=gs.cuda)
    
    scene_kwargs = {
        "sim_options": gs.options.SimOptions(
            dt=0.01,
            gravity=[0, 0, -9.81],
        ),
        "show_viewer": True
    }
    scene = gs.Scene(**scene_kwargs)
    
    # 地面を追加
    ground = scene.add_entity(gs.morphs.Plane())
    
    # HRP2 ロボットを追加
    print("[2/5] Loading HRP2 URDF...")
    robot_path = os.path.join(os.path.dirname(__file__), "hrp2_description/HRP2_genesis.urdf")
    try:
        robot = scene.add_entity(
            gs.morphs.URDF(file=robot_path, fixed=False)
        )
        print(f"✓ Successfully loaded HRP2 robot")
    except Exception as e:
        print(f"✗ Error loading URDF: {e}")
        return
    
    scene.build()
    
    # 制御対象の全関節を定義
    print("[3/5] Setting up full body control...")
    joint_names = [
        # 左脚 (6 DOF)
        "LLEG_JOINT0", "LLEG_JOINT1", "LLEG_JOINT2", "LLEG_JOINT3", "LLEG_JOINT4", "LLEG_JOINT5",
        # 右脚 (6 DOF)
        "RLEG_JOINT0", "RLEG_JOINT1", "RLEG_JOINT2", "RLEG_JOINT3", "RLEG_JOINT4", "RLEG_JOINT5",
        # 腰 (2 DOF)
        "CHEST_JOINT0", "CHEST_JOINT1",
        # 頭 (2 DOF)
        "HEAD_JOINT0", "HEAD_JOINT1",
        # 左腕 (7 DOF)
        "LARM_JOINT0", "LARM_JOINT1", "LARM_JOINT2", "LARM_JOINT3", "LARM_JOINT4", "LARM_JOINT5", "LARM_JOINT6",
        # 右腕 (7 DOF)
        "RARM_JOINT0", "RARM_JOINT1", "RARM_JOINT2", "RARM_JOINT3", "RARM_JOINT4", "RARM_JOINT5", "RARM_JOINT6",
    ]
    
    # 関節インデックスを取得
    dofs_idx_local = []
    for joint_name in joint_names:
        try:
            dof_idx = robot.get_joint(joint_name).dof_idx_local
            dofs_idx_local.append(dof_idx)
            print(f"  {joint_name}: dof_idx_local = {dof_idx}")
        except Exception as e:
            print(f"  Warning: Could not get joint {joint_name}: {e}")
    
    dofs_idx_local = np.array(dofs_idx_local)
    num_motors = len(dofs_idx_local)
    print(f"\n✓ Found {num_motors} motor DOFs\n")
    
    # PD制御パラメータを設定
    print("[4/5] Configuring PD control...")
    kp_values = np.array([
        # 左脚 (6 DOF)
        500.0, 500.0, 300.0, 500.0, 300.0, 300.0,
        # 右脚 (6 DOF)
        500.0, 500.0, 300.0, 500.0, 300.0, 300.0,
        # 腰 (2 DOF)
        300.0, 300.0,
        # 頭 (2 DOF)
        100.0, 100.0,
        # 左腕 (7 DOF)
        200.0, 200.0, 150.0, 150.0, 100.0, 100.0, 100.0,
        # 右腕 (7 DOF)
        200.0, 200.0, 150.0, 150.0, 100.0, 100.0, 100.0,
    ], dtype=np.float32)
    
    kv_values = np.array([
        # 左脚 (6 DOF)
        20.0, 20.0, 10.0, 20.0, 10.0, 10.0,
        # 右脚 (6 DOF)
        20.0, 20.0, 10.0, 20.0, 10.0, 10.0,
        # 腰 (2 DOF)
        15.0, 15.0,
        # 頭 (2 DOF)
        5.0, 5.0,
        # 左腕 (7 DOF)
        10.0, 10.0, 8.0, 8.0, 5.0, 5.0, 5.0,
        # 右腕 (7 DOF)
        10.0, 10.0, 8.0, 8.0, 5.0, 5.0, 5.0,
    ], dtype=np.float32)
    
    robot.set_dofs_kp(kp_values, dofs_idx_local)
    robot.set_dofs_kv(kv_values, dofs_idx_local)
    
    force_upper = np.array([
        # 左脚 (6 DOF)
        200.0, 200.0, 200.0, 200.0, 100.0, 100.0,
        # 右脚 (6 DOF)
        200.0, 200.0, 200.0, 200.0, 100.0, 100.0,
        # 腰 (2 DOF)
        100.0, 100.0,
        # 頭 (2 DOF)
        50.0, 50.0,
        # 左腕 (7 DOF)
        80.0, 80.0, 60.0, 60.0, 40.0, 40.0, 40.0,
        # 右腕 (7 DOF)
        80.0, 80.0, 60.0, 60.0, 40.0, 40.0, 40.0,
    ], dtype=np.float32)
    force_lower = -force_upper
    robot.set_dofs_force_range(force_lower, force_upper, dofs_idx_local)
    print("✓ PD control configured\n")
    
    # 初期姿勢を設定
    print("[5/5] Setting initial pose...")
    initial_angles = np.array([
        # 左脚: 立ち姿勢
        0.0, -0.2, 0.0, 0.4, -0.2, 0.0,
        # 右脚: 立ち姿勢
        0.0, -0.2, 0.0, 0.4, -0.2, 0.0,
        # 腰: 直立
        0.0, 0.0,
        # 頭: 正面向き
        0.0, 0.0,
        # 左腕: 垂直に下ろす
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # 右腕: 垂直に下ろす
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    
    for step in range(50):
        robot.set_dofs_position(initial_angles, dofs_idx_local)
        scene.step()
    
    # 落ち着かせる
    for _ in range(100):
        scene.step()
    
    print("✓ Initial pose set\n")
    
    # シミュレーション実行（全身制御の確認）
    print("="*70)
    print("Simulation Running (20 seconds)...")
    print("Observe:")
    print("  - Arms hang down without swaying")
    print("  - Head faces forward")
    print("  - Waist remains upright")
    print("  - Legs hold standing posture")
    print("="*70 + "\n")
    
    simulation_time = 20.0
    dt = 0.01
    num_steps = int(simulation_time / dt)
    
    for i in range(num_steps):
        # 全身の目標姿勢を保持（ブラブラ揺れを防止）
        target_angles = initial_angles.copy()
        robot.control_dofs_position(target_angles, dofs_idx_local)
        
        scene.step()
        
        if i % 500 == 0:
            elapsed = i * dt
            print(f"Time: {elapsed:.1f}s - All joints holding target position")
    
    print(f"\n✓ Simulation completed!")
    print(f"Total simulation time: {simulation_time}s")
    print("\nFull body control test finished.")
    print("If you see the robot standing still without swaying,")
    print("the full body control is working correctly!")


if __name__ == "__main__":
    test_full_body_control()
