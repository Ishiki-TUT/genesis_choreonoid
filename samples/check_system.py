#!/usr/bin/env python3
"""
HRP2 Walk Score システム - 準備確認スクリプト

このスクリプトでシステムが正しくセットアップされているか確認できます。
"""

import os
import sys
import pickle

def check_environment():
    """環境が正しくセットアップされているか確認"""
    print("\n" + "="*80)
    print("HRP2 WALK SCORE SYSTEM - ENVIRONMENT CHECK")
    print("="*80 + "\n")
    
    checks = []
    
    # 1. 必要なファイルの確認
    print("[1] Checking required files...")
    required_files = [
        "hrp2_train.py",
        "hrp2_eval_gs.py",
        "hrp2_env_gs.py",
        "hrp2_auto_train_loop.py",
        "test_walk_score.py",
        "WALK_SCORE_GUIDE.md",
        "QUICKSTART.md",
    ]
    
    for f in required_files:
        path = os.path.join(os.path.dirname(__file__), f)
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        checks.append(("File: " + f, exists))
    
    # 2. Python パッケージの確認
    print("\n[2] Checking Python packages...")
    packages = ["torch", "numpy", "genesis", "rsl_rl"]
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
            checks.append(("Package: " + pkg, True))
        except ImportError:
            print(f"  ❌ {pkg} (not installed)")
            checks.append(("Package: " + pkg, False))
    
    # 3. ロボットデータの確認
    print("\n[3] Checking robot files...")
    robot_files = [
        "hrp2_description/HRP2_genesis.urdf",
        "../irsl_rl/rl_env_base.py",
        "../irsl_rl/rl_env_gs.py",
    ]
    
    for f in robot_files:
        path = os.path.join(os.path.dirname(__file__), f)
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        checks.append(("File: " + f, exists))
    
    # 4. ログディレクトリ
    print("\n[4] Checking log directories...")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if os.path.exists(log_dir):
        num_exps = len([d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))])
        print(f"  ✅ logs/ ({num_exps} existing experiments)")
        checks.append(("Log directory", True))
    else:
        print(f"  ℹ️  logs/ (will be created automatically)")
        checks.append(("Log directory", True))
    
    # 5. 結果表示
    print("\n" + "="*80)
    total = len(checks)
    passed = sum(1 for _, result in checks if result)
    
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("\nSystem is ready! Run:")
        print("  python3 hrp2_auto_train_loop.py -e hrp2-walking --num_loops 5")
    else:
        print(f"⚠️  {total - passed} CHECK(S) FAILED ({passed}/{total})")
        print("\nFailed items:")
        for name, result in checks:
            if not result:
                print(f"  ❌ {name}")
        print("\nPlease install missing dependencies or download files.")
    
    print("="*80 + "\n")
    
    return passed == total

def print_system_overview():
    """システム概要を表示"""
    print("\n" + "="*80)
    print("HRP2 WALK SCORE SYSTEM - OVERVIEW")
    print("="*80 + "\n")
    
    print("""
【システムの役割】

  obs_data (観測データ) から、ロボットが「左右の足が交互に地面から離れる歩行」
  を実現できているかを自動判定し、リワード設定を最適化します。

【主要な指標】

  1. Air Time Score (35%)
     → 足が地面から離れている時間の割合
     → 目標: 30-50%

  2. Alternating Gait Score (35%)
     → 左足と右足が交互に出ているか
     → シャッフル（両足同時接地）を検出してペナルティ

  3. Speed Stability Score (20%)
     → 速度の追従度と安定性

  4. Symmetry Score (10%)
     → 左右の足の動きの対称性

【判定に使用するデータ】

  左足の高さ (left_foot_z)
  右足の高さ (right_foot_z)
    ↓
  高さ > 0.01m → 地面から離れている
  高さ ≤ 0.01m → 地面に接触している
    ↓
  足の状態遷移を解析
  → 交互パターンを判定
  → シャッフル期間を検出
  → スコア計算

【実行フロー】

  Training (100反復)
    ↓ 学習完了
  Evaluation (500ステップ)
    ↓ obs_data を記録
  Walk Score Calculation
    ↓ スコア: 0.0 ～ 1.0
  Reward Adjustment
    ↓ リワード値を調整
  (ループを繰り返し)

【出力例】

  Loop 1: Score = 0.35
    → Air Time: 20%, Alternating: 40%
    → Action: feet_air_time × 1.2

  Loop 2: Score = 0.55
    → Air Time: 35%, Alternating: 65%
    → Action: feet_alternating_pos × 1.15

  Loop 3: Score = 0.72
    → Air Time: 42%, Alternating: 80%
    → (良好なパターン形成)

  ... (Loop 4, 5を経て目標達成)

【使用方法】

  # 自動トレーニングループを実行（推奨）
  python3 hrp2_auto_train_loop.py -e hrp2-walking --num_loops 5

  # 単一実験のスコアを確認
  python3 test_walk_score.py -e hrp2-walking_loop1 --verbose

【ドキュメント】

  QUICKSTART.md         : 使い方ガイド
  WALK_SCORE_GUIDE.md   : 詳細な説明書

    """)
    
    print("="*80 + "\n")

if __name__ == "__main__":
    # システム概要を表示
    print_system_overview()
    
    # 環境チェック
    success = check_environment()
    
    sys.exit(0 if success else 1)
