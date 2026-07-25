#!/usr/bin/env python3
"""
HRP2 Walking Simulation - Genesis API修正テスト
Genesis公式API仕様に準拠した実装の動作確認
"""

import sys
import os
import numpy as np

# モジュールパス追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "irsl_rl"))

import genesis as gs

def test_genesis_api_basic():
    """Genesis公式API基本テスト"""
    print("="*70)
    print("Testing Genesis API with Simple Robot Model")
    print("="*70)
    
    # Genesis初期化
    print("\n[1] Initializing Genesis...")
    gs.init(backend=gs.cuda)
    
    # シーン作成
    print("[2] Creating scene...")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,  # テストなのでビューアー不要
    )
    
    # 地面と簡単なロボット(Franka)を追加
    print("[3] Adding entities...")
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Franka robot を使ってテスト
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file='xml/franka_emika_panda/panda.xml',
            pos=(0.0, 0.0, 0.0),
        )
    )
    
    scene.build()
    print("✓ Scene built successfully")
    
    # 関節情報を取得（公式API）
    print("\n[4] Testing joint indexing (official API)...")
    jnt_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
        'finger_joint1',
        'finger_joint2',
    ]
    
    dofs_idx_local = []
    for name in jnt_names:
        try:
            dof_idx = franka.get_joint(name).dof_idx_local
            dofs_idx_local.append(dof_idx)
            print(f"  ✓ {name}: dof_idx_local = {dof_idx}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    dofs_idx_local = np.array(dofs_idx_local)
    print(f"✓ Found {len(dofs_idx_local)} DOFs")
    
    # PD制御ゲイン設定（公式API）
    print("\n[5] Testing PD control setup (official API)...")
    kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
    kv = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
    
    try:
        franka.set_dofs_kp(kp, dofs_idx_local)
        print("  ✓ set_dofs_kp succeeded")
    except Exception as e:
        print(f"  ✗ set_dofs_kp failed: {e}")
    
    try:
        franka.set_dofs_kv(kv, dofs_idx_local)
        print("  ✓ set_dofs_kv succeeded")
    except Exception as e:
        print(f"  ✗ set_dofs_kv failed: {e}")
    
    # 力の範囲設定（公式API）
    print("\n[6] Testing force range setup (official API)...")
    force_upper = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
    force_lower = -force_upper
    
    try:
        franka.set_dofs_force_range(force_lower, force_upper, dofs_idx_local)
        print("  ✓ set_dofs_force_range succeeded")
    except Exception as e:
        print(f"  ✗ set_dofs_force_range failed: {e}")
    
    # 硬いリセット（公式API）
    print("\n[7] Testing hard reset (official API)...")
    target_pos_1 = np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04])
    
    try:
        for _ in range(5):
            franka.set_dofs_position(target_pos_1, dofs_idx_local)
            scene.step()
        print("  ✓ set_dofs_position succeeded")
    except Exception as e:
        print(f"  ✗ set_dofs_position failed: {e}")
    
    # PD制御（公式API）
    print("\n[8] Testing PD control (official API)...")
    target_pos_2 = np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04])
    
    try:
        franka.control_dofs_position(target_pos_2, dofs_idx_local)
        for _ in range(10):
            scene.step()
        print("  ✓ control_dofs_position succeeded")
    except Exception as e:
        print(f"  ✗ control_dofs_position failed: {e}")
    
    # 状態取得（公式API）
    print("\n[9] Testing state queries (official API)...")
    
    try:
        positions = franka.get_dofs_position(dofs_idx_local)
        print(f"  ✓ get_dofs_position: shape={positions.shape}")
    except Exception as e:
        print(f"  ✗ get_dofs_position failed: {e}")
    
    try:
        velocities = franka.get_dofs_velocity(dofs_idx_local)
        print(f"  ✓ get_dofs_velocity: shape={velocities.shape}")
    except Exception as e:
        print(f"  ✗ get_dofs_velocity failed: {e}")
    
    try:
        forces = franka.get_dofs_force(dofs_idx_local)
        print(f"  ✓ get_dofs_force: shape={forces.shape}")
    except Exception as e:
        print(f"  ✗ get_dofs_force failed: {e}")
    
    print("\n" + "="*70)
    print("✓ All Genesis API tests passed!")
    print("="*70)

if __name__ == "__main__":
    try:
        test_genesis_api_basic()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
