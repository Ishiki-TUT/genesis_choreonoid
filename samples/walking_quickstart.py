#!/usr/bin/env python3
"""
HRP2 Walking Plan Quick Start Script
さまざまなシナリオで歩行計画を生成・実行する便利なスクリプト
"""

import sys
import os
import subprocess
import argparse


def run_command(cmd, description):
    """コマンドを実行して結果を表示"""
    print("\n" + "="*70)
    print(f"🚶 {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def scenario_basic():
    """基本的な歩行計画"""
    cmd = [
        "python3", "hrp2_footplane.py",
        "--stride", "0.2",
        "--spacing", "0.2",
        "--com-height", "0.8",
        "--step-duration", "0.8",
        "--verbose"
    ]
    run_command(cmd, "Basic Walking Plan Generation")


def scenario_slow():
    """ゆっくり歩く"""
    cmd = [
        "python3", "hrp2_footplane.py",
        "--stride", "0.15",
        "--spacing", "0.2",
        "--step-duration", "1.0",
        "--com-height", "0.75",
        "--verbose"
    ]
    run_command(cmd, "Slow Walking Plan (Small steps)")


def scenario_fast():
    """速く歩く"""
    cmd = [
        "python3", "hrp2_footplane.py",
        "--stride", "0.25",
        "--spacing", "0.22",
        "--step-duration", "0.6",
        "--com-height", "0.82",
        "--verbose"
    ]
    run_command(cmd, "Fast Walking Plan (Large steps)")


def scenario_sidewalk():
    """横向き歩行"""
    cmd = [
        "python3", "hrp2_footplane.py",
        "--stride", "0.0",
        "--sway", "0.2",
        "--spacing", "0.25",
        "--step-duration", "0.8",
        "--verbose"
    ]
    run_command(cmd, "Sidewalk (Lateral movement)")


def scenario_turn():
    """回転歩行"""
    cmd = [
        "python3", "hrp2_footplane.py",
        "--stride", "0.2",
        "--turn", "0.1",
        "--spacing", "0.2",
        "--step-duration", "0.8",
        "--verbose"
    ]
    run_command(cmd, "Turning Walk")


def scenario_visualize():
    """Genesis での可視化"""
    cmd = [
        "python3", "hrp2_walking_visualizer.py",
        "--stride", "0.2",
        "--duration", "10",
        "--dt", "0.01"
    ]
    run_command(cmd, "Genesis Visualization (10 sec)")


def scenario_visualize_fast():
    """高速可視化"""
    cmd = [
        "python3", "hrp2_walking_visualizer.py",
        "--stride", "0.25",
        "--duration", "15",
        "--dt", "0.01",
        "--quiet"
    ]
    run_command(cmd, "Genesis Visualization - Fast (No verbose output)")


def scenario_visualize_no_viewer():
    """ビューアーなし（高速）"""
    cmd = [
        "python3", "hrp2_walking_visualizer.py",
        "--stride", "0.2",
        "--duration", "20",
        "--no-viewer",
        "--quiet"
    ]
    run_command(cmd, "Headless Simulation (No viewer, fastest)")


def scenario_save_trajectory():
    """軌跡を保存"""
    cmd = [
        "python3", "hrp2_walking_visualizer.py",
        "--stride", "0.2",
        "--duration", "5",
        "--save-trajectory", "trajectory_data.json",
        "--quiet"
    ]
    run_command(cmd, "Save Trajectory to JSON File")


def show_menu():
    """メニューを表示"""
    print("\n" + "="*70)
    print("HRP2 Walking Plan - Quick Start Scenarios")
    print("="*70)
    print("\n📋 Available scenarios:\n")
    
    scenarios = [
        ("1", "Basic Walking Plan (standard parameters)", scenario_basic),
        ("2", "Slow Walk (small steps, small stride)", scenario_slow),
        ("3", "Fast Walk (large steps, large stride)", scenario_fast),
        ("4", "Sidewalk (lateral movement)", scenario_sidewalk),
        ("5", "Turning Walk (with rotation)", scenario_turn),
        ("6", "Genesis Visualization (with viewer)", scenario_visualize),
        ("7", "Fast Visualization (no verbose output)", scenario_visualize_fast),
        ("8", "Headless Simulation (fastest, no viewer)", scenario_visualize_no_viewer),
        ("9", "Save Walking Trajectory to JSON", scenario_save_trajectory),
        ("0", "Exit", None),
    ]
    
    for num, desc, _ in scenarios:
        print(f"  [{num}] {desc}")
    
    print()
    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description='HRP2 Walking Plan - Quick Start Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 walking_quickstart.py                    # Interactive menu
  python3 walking_quickstart.py --scenario 1       # Run scenario 1
  python3 walking_quickstart.py --scenario all     # Run all scenarios
  python3 walking_quickstart.py --scenario basic   # Run by name
        """
    )
    
    parser.add_argument('--scenario', type=str, default=None,
                       help='Scenario number or name (or "all" to run all)')
    parser.add_argument('--no-menu', action='store_true',
                       help='Skip interactive menu')
    
    args = parser.parse_args()
    
    scenarios_list = [
        ("1", "basic", scenario_basic),
        ("2", "slow", scenario_slow),
        ("3", "fast", scenario_fast),
        ("4", "sidewalk", scenario_sidewalk),
        ("5", "turn", scenario_turn),
        ("6", "visualize", scenario_visualize),
        ("7", "visualize_fast", scenario_visualize_fast),
        ("8", "headless", scenario_visualize_no_viewer),
        ("9", "save", scenario_save_trajectory),
    ]
    
    # コマンドライン引数処理
    if args.scenario:
        selected_scenarios = []
        
        if args.scenario.lower() == "all":
            selected_scenarios = scenarios_list
        else:
            # 番号または名前でマッチング
            for num, name, func in scenarios_list:
                if args.scenario in [num, name]:
                    selected_scenarios.append((num, name, func))
                    break
        
        if not selected_scenarios:
            print(f"✗ Scenario '{args.scenario}' not found")
            return 1
        
        # 選択されたシナリオを実行
        for num, name, func in selected_scenarios:
            try:
                func()
            except KeyboardInterrupt:
                print("\n\n⊘ Cancelled by user")
                return 1
            except Exception as e:
                print(f"✗ Error: {e}")
                return 1
        
        return 0
    
    # インタラクティブメニュー
    print("\n" + "="*70)
    print("HRP2 Walking Plan - Quick Start")
    print("="*70)
    
    while True:
        scenarios_menu = show_menu()
        
        try:
            choice = input("Enter scenario number (0 to exit): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                return 0
            
            # メニュー選択
            selected_func = None
            for num, desc, func in scenarios_menu:
                if choice == num and func is not None:
                    selected_func = func
                    break
            
            if selected_func is None:
                print("✗ Invalid choice. Please try again.")
                continue
            
            # シナリオを実行
            try:
                selected_func()
                print("\n✓ Scenario completed successfully!")
            except KeyboardInterrupt:
                print("\n\n⊘ Scenario cancelled by user")
                continue
            except Exception as e:
                print(f"\n✗ Error during scenario: {e}")
                continue
            
            # 続行確認
            again = input("\nRun another scenario? (y/n) [y]: ").strip().lower()
            if again in ['n', 'no']:
                print("👋 Goodbye!")
                return 0
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            return 0
        except Exception as e:
            print(f"✗ Error: {e}")
            continue


if __name__ == "__main__":
    sys.exit(main())
