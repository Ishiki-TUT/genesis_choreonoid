# HRP2 歩行最適化システム - Walk Score による自動リワード調整

## 概要

このシステムは、**シミュレーション内のロボット（HRP2）の歩行動作データ（obs_data）から、「左右の足が交互に地面から離れる歩行（No Shuffle Walking）」を実現できているかを自動判定し、リワード設定を動的に最適化**します。

### 核となる質問への回答

> **"obs_data から目標とする歩行になっているか確認するやり方は？"**

**答え**: 以下の4つの指標を obs_data から計算して判定します：

```
目標判定 = 
  35% × 足の滞空時間スコア
  + 35% × 足の交互性スコア  
  + 20% × 速度安定性スコア
  + 10% × 左右対称性スコア
  = Walk Score (0.0 ～ 1.0)
```

---

## 判定方法の詳細

### 1. **足の滞空時間スコア** (Air Time Score)

obs_data の **左足高さ (left_foot_z)** と **右足高さ (right_foot_z)** から判定：

```python
判定基準:
  高さ > 0.01m (10mm)  → 地面から離れている（浮遊）
  高さ ≤ 0.01m         → 地面に接触している（接地）

スコア計算:
  左足滞空比 = (left_foot_z > 0.01).mean()
  右足滞空比 = (right_foot_z > 0.01).mean()
  
  平均滞空比が 30-50% で高スコア
```

### 2. **足の交互性スコア** (Alternating Gait Score) ← 最重要

obs_data から足の接地状態の**遷移パターン**を解析：

```python
step-by-step:
  
  1. 足の接地状態を判定
     left_contact  = (left_foot_z ≤ 0.01)   # [0,1,0,1,0,1,...]
     right_contact = (right_foot_z ≤ 0.01)  # [1,0,1,0,1,0,...]
  
  2. 両足が同時に接地している時間を検出
     both_contact = left_contact & right_contact
     simultaneous_ratio = both_contact.mean()  # 低いほど良い
  
  3. 「シャッフル」を検出
     5ステップ以上連続して両足が接地 → ペナルティ
  
  4. 足の切り替えが規則的か判定
     L→R→L→R... のパターンがあるか
  
  5. スコア化
     Alternating Score = (交互性率) × (1-シャッフルペナルティ)
```

### 3. **速度安定性スコア** (Speed Stability Score)

obs_data の **速度 (lin_vel)** と **指令速度 (command_vel)** から判定：

```python
追従度  = 1.0 - |実速度 - 指令速度| / 指令速度
安定性  = 1.0 - (速度の標準偏差)/(速度の平均値)
  
Speed Stability = 60% × 追従度 + 40% × 安定性
```

### 4. **左右対称性スコア** (Symmetry Score)

左足と右足の動きが対称的か判定：

```python
高さの差  = |左足高さ - 右足高さ|.mean()
滞空バランス = 左足滞空時間 / 右足滞空時間 の比

Symmetry Score = 高さ対称性 × 0.6 + 滞空バランス × 0.4
```

---

## システムの実行フロー

```
┌─────────────────────────────────────────────────────────┐
│ AUTO-TRAINING LOOP (5ループの例)                         │
└─────────────────────────────────────────────────────────┘

Loop 1:
  ├─ Training (100反復)
  │   └─ ポリシーを学習（初期リワード）
  ├─ Evaluation (500ステップ)
  │   └─ obs_data を記録
  │       (left_foot_z, right_foot_z, lin_vel, command_vel, ...)
  ├─ Walk Score Calculation
  │   ├─ Air Time Score = 0.50
  │   ├─ Alternating Score = 0.40
  │   ├─ Speed Stability = 0.60
  │   ├─ Symmetry Score = 0.65
  │   └─ TOTAL SCORE = 0.48 ⚠️
  └─ Reward Adjustment
      └─ Score < 0.6 → feet_alternating_pos × 1.15

Loop 2:
  ├─ Training (100反復) ← 調整されたリワードで学習
  ├─ Evaluation
  └─ Score = 0.62 ✅ (改善！)

... (Loop 3, 4, 5を繰り返し)

Loop 5:
  └─ Score = 0.82 🎉 (目標達成！)
```

---

## 実行コマンド

### 1. 自動トレーニングループ（推奨）

```bash
cd samples
python3 hrp2_auto_train_loop.py \
    -e hrp2-walking \
    --max_iterations 100 \
    --num_loops 5
```

**出力**:
```
Loop 1/5
Running: python3 hrp2_train.py -e hrp2-walking_loop1 --max_iterations 100
Training completed.

Evaluating: hrp2-walking_loop1
Running: python3 hrp2_eval_gs.py -e hrp2-walking_loop1 --ckpt 100

Walk Quality Metrics for hrp2-walking_loop1:
────────────────────────────────────────────
Air Time Score:          0.5843
Alternating Score:       0.4123
Speed Stability Score:   0.6234
Symmetry Score:          0.6512
────────────────────────────────────────────
TOTAL SCORE:             0.5603 ⚠️

Adjusting rewards based on score: 0.5603
...
```

### 2. 単一実験のスコアを詳細に確認

```bash
python3 test_walk_score.py -e hrp2-walking_loop1 --verbose
```

**出力**:
```
GAIT PATTERN VISUALIZATION
──────────────────────────
Pattern (every 5 steps):
LxRxLxRxLxRxLxRxLxRxLx...

Statistics:
  Left foot air time: 42.3%
  Right foot air time: 41.2%
  Both feet in air: 12.5%
  Both feet on ground: 5.2% (シャッフルほぼなし！)

WALK SCORE CALCULATION
──────────────────────
Air Time Score:     0.85 ✅
Alternating Score:  0.82 ✅
Speed Stability:    0.78 ✅
Symmetry Score:     0.91 ✅

FINAL WALK SCORE: 0.84 🎉

Score Interpretation:
  ✅ Good - Clear alternating pattern with minimal shuffling
     Action: Fine-tune for speed and stability
```

---

## スコア別の推奨アクション

| スコア | 解釈 | 推奨アクション |
|--------|------|----------------|
| < 0.3 | 不良 | `feet_air_time × 1.2`<br>`feet_alternating_pos × 1.1` |
| 0.3～0.6 | 中程度 | `feet_alternating_pos × 1.15`<br>`ankle_regularization × 1.1` |
| 0.6～0.8 | 良好 | `orientation × 1.05`<br>`action_rate × 0.95` |
| ≥ 0.8 | 優秀 | `ankle_regularization × 1.2`<br>`orientation × 1.1` |

---

## 出力ファイル

```
auto_train_logs/YYYYMMDD_HHMMSS/
├─ final_results.json      ← 最終結果サマリー
├─ rewards_loop1.json      ← Loop 1のリワード設定
├─ rewards_loop2.json
└─ ...

logs/hrp2-walking_loop1/
├─ eval_metrics.pkl        ← 評価メトリクス（obs_data）
├─ model_*.pt              ← 学習済みモデル
├─ cfgs.pkl                ← 環境設定
└─ ...
```

---

## eval_metrics.pkl の内容

```python
metrics = {
    'left_foot_z': array([...]),      # 左足の高さ (num_steps,)
    'right_foot_z': array([...]),     # 右足の高さ (num_steps,)
    'lin_vel': array([...]),          # 前進速度 (num_steps,)
    'command_vel': array([...]),      # 指令速度 (num_steps,)
    'base_height': array([...]),      # ベース高さ (num_steps,)
    'base_roll': array([...]),        # ロール角度 (num_steps,)
    'base_pitch': array([...]),       # ピッチ角度 (num_steps,)
    'action_magnitude': array([...]), # アクション大きさ (num_steps,)
    'rewards': array([...]),          # ステップ報酬 (num_steps,)
}
```

---

## トラブルシューティング

### Q: スコアが常に 0.0

```
A: eval_metrics.pkl が正しく生成されているか確認:

  1. ファイルの存在確認:
     $ ls -la logs/<exp_name>/eval_metrics.pkl
  
  2. hrp2_eval_gs.py の出力に足の高さが記録されているか確認:
     $ grep "foot avg height" logs/<exp_name>/output.log
  
  3. hrp2_env_gs.py の specific_update_buffer() が実装されているか確認
```

### Q: スコアが改善しない

```
A: 詳細スコアを確認して、どの要素が弱いか特定:

  python3 test_walk_score.py -e <exp_name> --verbose

  もしも Air Time が低い場合:
    feet_air_time リワードを増加
  
  もしも Alternating が低い場合:
    feet_alternating_pos リワードを増加
```

---

## ドキュメント

- **QUICKSTART.md**: 実行方法ガイド
- **WALK_SCORE_GUIDE.md**: 詳細な説明書（各スコア要素の詳細）
- **check_system.py**: システム準備状況を確認

---

## 関連ファイル

| ファイル | 役割 |
|---------|------|
| `hrp2_train.py` | ポリシーを学習 |
| `hrp2_eval_gs.py` | ポリシーを評価、メトリクスを記録 |
| `hrp2_env_gs.py` | シミュレーション環境、リワード関数定義 |
| `hrp2_auto_train_loop.py` | 自動最適化ループ（Main） |
| `test_walk_score.py` | スコア計算・可視化（デバッグ用） |

---

## 成功例

```
実験: hrp2-walking

Loop 1: Score = 0.35
  Air Time: 20%, Alternating: 40%, Speed: 0.50
  Action: feet_air_time × 1.2

Loop 2: Score = 0.55
  Air Time: 35%, Alternating: 65%, Speed: 0.65
  Action: feet_alternating_pos × 1.15

Loop 3: Score = 0.72 ✅
  Air Time: 42%, Alternating: 80%, Speed: 0.75
  Action: Minor adjustment

Loop 4: Score = 0.80 🎉
  Air Time: 45%, Alternating: 85%, Speed: 0.85

Loop 5: Score = 0.82 🎉
  Air Time: 48%, Alternating: 87%, Speed: 0.90

最終結果:
  ✅ 左右の足が交互に地面から離れる歩行が実現
  ✅ シャッフルなし（両足同時接地 < 5%）
  ✅ 速度追従性 90%
  ✅ 左右対称性 87%
```

---

## まとめ

このシステムの核となる判定ロジック：

1. **足の高さデータ** (left_foot_z, right_foot_z) を obs_data から抽出
2. **接地状態** (高さ > 0.01m) を判定
3. **交互パターン** を検出（L-R-L-R...）
4. **シャッフル期間** を検出して ペナルティ
5. **スコア計算** (4つの要素の加重平均)
6. **リワード調整** (スコアに基づいて)

このループを繰り返すことで、「左右の足が交互に地面から離れる歩行」を自動的に実現できます。

