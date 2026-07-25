#!/usr/bin/env python3
"""
HRP2 自動歩行最適化システム - クイックスタートガイド

このスクリプトは、hrp2_auto_train_loop.py の使用方法を示します。
"""

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           HRP2 自動歩行最適化システム - クイックスタート                      ║
╚════════════════════════════════════════════════════════════════════════════╝

【目標】
  左右の足が交互に地面から離れる歩行（No Shuffle Walking）を実現する

【システムの流れ】

  1. Training (hrp2_train.py)
         ↓ ポリシーを学習
  2. Evaluation (hrp2_eval_gs.py)
         ↓ 歩行性能を評価、メトリクスを記録
  3. Walk Score Calculation
         ↓ obs_data から歩行スコアを計算
  4. Reward Adjustment
         ↓ スコアに基づいてリワードを調整
  5. Loop (2-4を繰り返し)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【使い方】

【方法1】自動トレーニングループ（推奨）

  $ cd samples
  $ python3 hrp2_auto_train_loop.py \\
      -e hrp2-walking \\
      --max_iterations 100 \\
      --num_loops 5

  パラメータ:
    -e, --exp_name       : 実験名 (デフォルト: hrp2-walking)
    --max_iterations     : 各ループでのトレーニング反復回数 (デフォルト: 100)
    --num_loops          : 最適化ループ数 (デフォルト: 5)

  出力:
    - auto_train_logs/YYYYMMDD_HHMMSS/ : ログディレクトリ
    - logs/hrp2-walking_loop1/
    - logs/hrp2-walking_loop2/
    - ...
    - logs/hrp2-walking_loop5/

【方法2】単一の実験のスコアを確認

  $ python3 test_walk_score.py -e hrp2-walking_loop1 --verbose

  出力:
    - 歩行パターンの可視化
    - 各スコア要素の詳細
    - 総合スコアと推奨アクション

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【Walk Score の見方】

  スコア構成（4つの要素、重み付け合計=1.0）:

    ┌─────────────────────────────────────────────────┐
    │ 1. Air Time Score (35%)                         │
    │    → 足が地面から離れている時間の割合             │
    │    → 目標: 30-50% で高スコア                    │
    │                                                 │
    │ 2. Alternating Gait Score (35%)                │
    │    → 左足と右足が交互に出ているか                │
    │    → 目標: 交互性 80%以上                      │
    │    → シャッフル検出: 両足同時接地期間のペナルティ │
    │                                                 │
    │ 3. Speed Stability Score (20%)                 │
    │    → 指令速度への追従度と速度の安定性             │
    │    → 目標: スコア 0.7以上                      │
    │                                                 │
    │ 4. Symmetry Score (10%)                        │
    │    → 左足と右足の動きが対称的か                  │
    │    → 目標: スコア 0.7以上                      │
    └─────────────────────────────────────────────────┘

  総合スコア = 0.35×Air + 0.35×Alternating + 0.20×Speed + 0.10×Symmetry
           (0.0 ～ 1.0の値)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【スコア別の推奨アクション】

  ❌ スコア < 0.3: 不良（基本的な歩行ができていない）
     → feet_air_time リワードを 1.2倍増加
     → feet_alternating_pos リワードを 1.1倍増加

  ⚠️  0.3 ≤ スコア < 0.6: 中程度（交互パターンが見られるが不安定）
     → feet_alternating_pos リワードを 1.15倍増加
     → ankle_regularization を 1.1倍増加

  ✅ 0.6 ≤ スコア < 0.8: 良好（明確な交互パターン形成）
     → orientation リワードを 1.05倍増加
     → action_rate を 0.95倍減少（平滑化）

  🎉 スコア ≥ 0.8: 優秀（良好な交互歩行実現）
     → ankle_regularization を 1.2倍増加（精密化）
     → orientation を 1.1倍増加

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【トレーニング進行例】

  Loop 1: Score = 0.35 ⚠️
    Air Time: 20%, Alternating: 40%, Speed: 0.50
    → Action: feet_air_time × 1.2

  Loop 2: Score = 0.55 ⚠️
    Air Time: 35%, Alternating: 65%, Speed: 0.65
    → Action: feet_alternating_pos × 1.15

  Loop 3: Score = 0.72 ✅
    Air Time: 42%, Alternating: 80%, Speed: 0.75
    → Action: orientation × 1.05

  Loop 4: Score = 0.80 🎉
    Air Time: 45%, Alternating: 85%, Speed: 0.85
    → Action: Fine-tune only

  Loop 5: Score = 0.82 🎉
    Air Time: 48%, Alternating: 87%, Speed: 0.90
    → Goal Achieved!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【出力ファイル】

  auto_train_logs/YYYYMMDD_HHMMSS/
    ├─ final_results.json          : 最終結果サマリー
    ├─ rewards_loop1.json          : Loop 1のリワード設定
    ├─ rewards_loop2.json          : Loop 2のリワード設定
    └─ ...

  logs/hrp2-walking_loop1/
    ├─ model_*.pt                  : 学習済みモデル
    ├─ eval_metrics.pkl            : 評価メトリクス（走行データ）
    ├─ cfgs.pkl                    : 環境設定
    └─ ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【キーファイルの役割】

  hrp2_train.py
    → ポリシーを学習（リワードに基づいて最適化）

  hrp2_eval_gs.py
    → 学習済みポリシーを評価
    → obs_data から foot height, velocity etc. を記録
    → eval_metrics.pkl に保存

  hrp2_env_gs.py
    → シミュレーション環境の定義
    → 足の高さ計算: get_link_lowest_point_z()
    → リワード関数定義

  hrp2_auto_train_loop.py
    → 自動最適化ループの実装
    → calculate_walk_score() で obs_data を解析
    → adjust_rewards() でリワードを調整

  test_walk_score.py
    → Walk Score を単独で計算・可視化
    → デバッグ用

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【トラブルシューティング】

【Q】スコアが常に0.0
【A】
  1. eval_metrics.pkl が生成されているか確認:
     $ ls -la logs/<exp_name>/eval_metrics.pkl

  2. hrp2_eval_gs.py の出力に foot height が記録されているか確認

  3. hrp2_env_gs.py の specific_update_buffer() が正しく実装されているか確認

【Q】スコアが低い/改善しない
【A】
  1. リワード値が極端になっていないか確認:
     $ cat auto_train_logs/YYYYMMDD_HHMMSS/rewards_loopN.json

  2. 単一実験のスコアを詳細に確認:
     $ python3 test_walk_score.py -e <exp_name> --verbose

  3. どのスコア要素が低いか特定し、対応するリワードを調整

【Q】環境がクラッシュする
【A】
  1. Genesis のバージョンを確認:
     $ python3 -c "import genesis as gs; print(gs.__version__)"

  2. ロボットのURDFパスが正しいか確認:
     $ ls -la samples/hrp2_description/HRP2_genesis.urdf

  3. リンク名が正しいか確認:
     $ grep -E "LLEG_LINK5|RLEG_LINK5" samples/hrp2_description/HRP2_genesis.urdf

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【詳細なドキュメント】

  WALK_SCORE_GUIDE.md を参照してください
  $ cat WALK_SCORE_GUIDE.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

if __name__ == "__main__":
    main()
