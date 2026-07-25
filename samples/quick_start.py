#!/usr/bin/env python3
"""
HRP2 Walking Simulation - Quick Start Guide
3つのシンプルな実行方法を提供
"""

import subprocess
import sys
import os

def run_scenario(name, description, command):
    """シナリオを実行"""
    print("\n" + "="*70)
    print(f"🚶 {description}")
    print("="*70)
    print(f"Command: {command}\n")
    
    try:
        result = subprocess.run(command, shell=True, cwd=os.path.dirname(__file__))
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⊘ Cancelled")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("HRP2 Walking Simulation - 3 Simple Methods")
    print("="*70)
    
    print("\n📋 Available Methods:\n")
    print("  [1] Generate walking plan only (no visualization)")
    print("  [2] Simple Genesis visualization")
    print("  [3] Advanced visualization with RL environment")
    print("  [0] Exit")
    print()
    
    scenarios = {
        "1": (
            "Walking Plan Generation",
            "Generate 5-step deterministic walking plan",
            "python3 hrp2_footplane.py --verbose --duration 10"
        ),
        "2": (
            "Simple Genesis Visualization",
            "Run HRP2 walking in Genesis simulator (recommended for first-time use)",
            "python3 hrp2_walking_simple.py --duration 10"
        ),
        "3": (
            "Advanced RL Environment Visualization",
            "Use existing RL environment with walking plan",
            "python3 hrp2_walking_visualizer.py --duration 10"
        ),
    }
    
    while True:
        try:
            choice = input("Select method (0-3): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                return 0
            
            if choice not in scenarios:
                print("✗ Invalid choice. Please try again.\n")
                continue
            
            name, description, command = scenarios[choice]
            success = run_scenario(name, description, command)
            
            if not success:
                print("\n⚠ Scenario did not complete successfully")
            
            # Continue or exit
            again = input("\nRun another scenario? (y/n) [y]: ").strip().lower()
            if again in ['n', 'no']:
                print("\n👋 Goodbye!")
                return 0
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            return 0
        except Exception as e:
            print(f"Error: {e}\n")
            continue


if __name__ == "__main__":
    sys.exit(main())
