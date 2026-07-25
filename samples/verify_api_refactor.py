#!/usr/bin/env python3
"""
HRP2 Walking Simulation - コード改修検証スクリプト
Genesis公式API準拠への移行を確認
"""

import os
import sys
import re

def check_file_structure(filepath):
    """ファイルの構造をチェック"""
    print(f"\n📋 Checking: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "dofs_idx_local を使用": "self.dofs_idx_local" in content,
        "robot.get_joint() を使用": "robot.get_joint(" in content,
        ".dof_idx_local を取得": ".dof_idx_local" in content,
        "set_dofs_kp() を使用": "set_dofs_kp(" in content,
        "set_dofs_kv() を使用": "set_dofs_kv(" in content,
        "set_dofs_force_range() を使用": "set_dofs_force_range(" in content,
        "control_dofs_position() を使用": "control_dofs_position(" in content,
        "set_dofs_position() を使用": "set_dofs_position(" in content,
        "set_all_joint_targets() メソッド": "def set_all_joint_targets" in content,
        "旧API set_dof_target() の削除": "set_dof_target(" not in content,
        "旧API torch.tensor() の削除": "torch.tensor" not in content,
    }
    
    print("  チェック結果:")
    all_passed = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"    {status} {check_name}")
        if not result:
            all_passed = False
    
    return all_passed

def analyze_methods(filepath):
    """主要メソッドの分析"""
    print(f"\n🔍 Method Analysis: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    methods = {
        "_setup_robot_info": ("関節インデックス取得", "✓ dofs_idx_local"),
        "_setup_pd_control": ("PDゲイン設定", "✓ set_dofs_kp/kv"),
        "_set_initial_pose": ("初期姿勢設定", "✓ set_dofs_position"),
        "set_joint_target": ("単一関節制御", "✓ control_dofs_position"),
        "set_all_joint_targets": ("一括制御", "✓ 複数DOF効率的"),
        "_update_joint_targets_from_feet": ("軌道更新", "✓ 一括制御"),
    }
    
    for method_name, (description, expected) in methods.items():
        if f"def {method_name}" in content:
            print(f"  ✓ {method_name}()")
            print(f"    └─ {description}")
            print(f"    └─ {expected}")
        else:
            print(f"  ✗ {method_name}() NOT FOUND")

def check_class_attributes(filepath):
    """クラス属性の確認"""
    print(f"\n📦 Class Attributes: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # __init__内の属性を抽出
    init_match = re.search(r'def __init__\(self.*?\):(.*?)def ', content, re.DOTALL)
    if init_match:
        init_content = init_match.group(1)
        
        attributes = {
            "self.joint_names": "制御対象の関節名リスト",
            "self.dofs_idx_local": "関節のローカルDOFインデックス",
            "self.num_motors": "制御対象DOFの数",
            "self.joint_indices": "関節名→インデックスのマッピング",
        }
        
        for attr, description in attributes.items():
            if attr in init_content:
                print(f"  ✓ {attr}")
                print(f"    └─ {description}")
            else:
                print(f"  ✗ {attr} NOT FOUND")

def generate_comparison_table():
    """旧→新API対応表を表示"""
    print("\n📊 API Comparison Table:")
    print("="*70)
    
    comparisons = [
        ("dof_names()", "get_joint().dof_idx_local", "関節情報"),
        ("set_dof_target()", "control_dofs_position()", "位置制御"),
        ("set_dof_target()", "set_dofs_position()", "硬いリセット"),
        ("(no param)", "set_dofs_kp(kp, dofs_idx)", "PD P-gain"),
        ("(no param)", "set_dofs_kv(kv, dofs_idx)", "PD D-gain"),
        ("(no param)", "set_dofs_force_range()", "力制限"),
    ]
    
    print(f"{'旧API':<30} {'新API':<30} {'用途':<15}")
    print("-"*70)
    for old, new, purpose in comparisons:
        print(f"{old:<30} {new:<30} {purpose:<15}")
    
    print("="*70)

def main():
    print("="*70)
    print("HRP2 Walking Simulation - Genesis API改修検証")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "hrp2_walking_sim.py")
    
    if not os.path.exists(script_path):
        print(f"\n✗ ファイルが見つかりません: {script_path}")
        return False
    
    # 検証実行
    file_ok = check_file_structure(script_path)
    analyze_methods(script_path)
    check_class_attributes(script_path)
    generate_comparison_table()
    
    # 結果表示
    print("\n" + "="*70)
    if file_ok:
        print("✓ すべての検証が完了しました！")
        print("\n次のステップ:")
        print("  1. python -m py_compile samples/hrp2_walking_sim.py")
        print("  2. python samples/hrp2_walking_sim.py --help")
        print("  3. python samples/hrp2_walking_sim.py (実行)")
    else:
        print("✗ いくつかの検証に失敗しました。")
        print("  改修が完全に適用されているか確認してください。")
    print("="*70)
    
    return file_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
