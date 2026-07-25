#!/usr/bin/env python3
"""
歩行スコア計算システムのテストスクリプト
obs_dataから目標とする歩行（左右交互に足が地面から離れる）を判定する

Usage:
    python3 test_walk_score.py -e <exp_name>
    
Example:
    python3 test_walk_score.py -e hrp2-walking_loop1
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch

# パス設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "irsl_rl"))

from hrp2_auto_train_loop import RewardOptimizer

def visualize_gait_pattern(metrics):
    """
    歩行パターンを可視化
    左足と右足の高さと接地状態を表示
    """
    if 'left_foot_z' not in metrics or 'right_foot_z' not in metrics:
        print("No foot height data available")
        return
    
    left_z = metrics['left_foot_z']
    right_z = metrics['right_foot_z']
    
    if isinstance(left_z, torch.Tensor):
        left_z = left_z.cpu().numpy()
    if isinstance(right_z, torch.Tensor):
        right_z = right_z.cpu().numpy()
    else:
        left_z = np.asarray(left_z)
        right_z = np.asarray(right_z)
    
    # 足が地面から離れているかを判定
    left_airborne = (left_z > 0.01).astype(int)
    right_airborne = (right_z > 0.01).astype(int)
    
    print("\n" + "="*80)
    print("GAIT PATTERN VISUALIZATION")
    print("="*80)
    print("L = Left foot in air (離地), l = Left foot on ground (着地)")
    print("R = Right foot in air (離地), r = Right foot on ground (着地)")
    print("X = Both feet in air (両足離地 - 望ましい), x = Both feet on ground (両足着地)")
    print()
    
    # サンプリング（全ステップを表示すると長すぎるので、100ステップごとに表示）
    sample_rate = max(1, len(left_z) // 200)  # 最大200文字
    
    pattern = ""
    for i in range(0, len(left_z), sample_rate):
        l_state = left_airborne[i]
        r_state = right_airborne[i]
        
        if l_state and r_state:
            pattern += "X"  # 両足在宅
        elif l_state and not r_state:
            pattern += "L"  # 左足在宅
        elif not l_state and r_state:
            pattern += "R"  # 右足在宅
        else:
            pattern += "x"  # 両足着地
    
    print("Pattern (every {} steps):".format(sample_rate))
    print(pattern)
    print()
    
    # 統計情報
    left_air = left_airborne.mean()
    right_air = right_airborne.mean()
    both_air = (left_airborne & right_airborne).sum() / len(left_z)
    both_ground = ((1 - left_airborne) & (1 - right_airborne)).sum() / len(left_z)
    
    print(f"Statistics:")
    print(f"  Left foot air time: {left_air:.1%}")
    print(f"  Right foot air time: {right_air:.1%}")
    print(f"  Both feet in air: {both_air:.1%} (両足在宅時間)")
    print(f"  Both feet on ground: {both_ground:.1%} (両足着地時間 - シャッフル判定基準)")
    print()
    
    # シャッフル検出
    both_ground_signal = (1 - left_airborne) * (1 - right_airborne)
    shuffle_periods = detect_shuffle_periods(both_ground_signal)
    
    print(f"Shuffle Analysis:")
    print(f"  Number of shuffle periods (5+ steps with both feet on ground): {len(shuffle_periods)}")
    if shuffle_periods:
        print(f"  Shuffle period lengths: {shuffle_periods[:10]}...")  # 最初の10個を表示
        print(f"  Average shuffle length: {np.mean(shuffle_periods):.1f} steps")
    print()
    
    # 交互性判定
    print("Alternating Pattern Analysis:")
    left_transitions = detect_transitions(left_airborne)
    right_transitions = detect_transitions(right_airborne)
    print(f"  Left foot transitions: {len(left_transitions)}")
    print(f"  Right foot transitions: {len(right_transitions)}")
    
    alternating_count = count_alternating_pattern(left_transitions, right_transitions)
    max_transitions = max(len(left_transitions), len(right_transitions))
    if max_transitions > 0:
        alternating_ratio = alternating_count / max_transitions
        print(f"  Alternating ratio: {alternating_ratio:.1%}")
    print()

def detect_transitions(signal):
    """接地状態の遷移を検出"""
    transitions = []
    for i in range(1, len(signal)):
        if signal[i] != signal[i-1]:
            transitions.append(i)
    return transitions

def detect_shuffle_periods(both_ground_signal):
    """両足が同時に接地し続ける期間を検出"""
    periods = []
    duration = 0
    for in_contact in both_ground_signal:
        if in_contact > 0.5:
            duration += 1
        else:
            if duration > 5:
                periods.append(duration)
            duration = 0
    if duration > 5:
        periods.append(duration)
    return periods

def count_alternating_pattern(left_trans, right_trans):
    """左右の足が交互に切り替わっているかカウント"""
    if not left_trans or not right_trans:
        return 0
    
    alternating = 0
    for i in range(min(len(left_trans), len(right_trans)) - 1):
        if i < len(left_trans) - 1 and i < len(right_trans):
            if (left_trans[i] < right_trans[i] < left_trans[i + 1]) or \
               (right_trans[i] < left_trans[i] < right_trans[i + 1]):
                alternating += 1
    return alternating

def main():
    parser = argparse.ArgumentParser(description="Test walk score calculation from evaluation metrics")
    parser.add_argument("-e", "--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # RewardOptimizerを使用してスコアを計算
    optimizer = RewardOptimizer()
    
    # メトリクスを読み込み
    print(f"Loading metrics for: {args.exp_name}")
    metrics = optimizer.get_evaluation_metrics(args.exp_name)
    
    if metrics is None:
        print(f"Error: Could not load metrics for {args.exp_name}")
        return
    
    # 歩行パターンを可視化
    visualize_gait_pattern(metrics)
    
    # スコア計算
    print("="*80)
    print("WALK SCORE CALCULATION")
    print("="*80)
    score = optimizer.calculate_walk_score(args.exp_name)
    
    print("\n" + "="*80)
    print(f"FINAL WALK SCORE: {score:.4f}")
    print("="*80)
    
    # スコアの解釈
    print("\nScore Interpretation:")
    if score < 0.3:
        print("  ❌ Poor - Robot is not walking properly")
        print("     Action: Increase feet_air_time and feet_alternating_pos rewards")
    elif score < 0.6:
        print("  ⚠️  Fair - Some alternating pattern but needs improvement")
        print("     Action: Focus on reducing simultaneous foot contact")
    elif score < 0.8:
        print("  ✅ Good - Clear alternating pattern with minimal shuffling")
        print("     Action: Fine-tune for speed and stability")
    else:
        print("  🎉 Excellent - Strong alternating gait achieved!")
        print("     Action: Minimal adjustment needed")

if __name__ == "__main__":
    main()
