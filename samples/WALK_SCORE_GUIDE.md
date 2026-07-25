# HRP2 歩行最適化システム - Walk Score の説明

## 概要

このドキュメントは、`hrp2_auto_train_loop.py` で実装される歩行スコア計算システムについて説明します。このシステムは、**左右の足が交互に地面から離れる歩行（No Shuffle Walking）を自動的に判定・最適化**します。

---

## 歩行目標

### 理想的な歩行パターン

```
時刻 ─────────────────────────────────────
左足 ─╭───┬─╭───┬─╭───...  (交互に上げ下げ)
     │   │ │   │ │
右足 ─┼───╭─┼───╭─┼───...  (左足と逆位相)
     │   │ │   │ │
```

- **滞空時間（Air Time）**: 両足が同時に地面から離れる時間 → 10-30%が目標
- **足の交互性**: 左足と右足が交互に出ている → 交互性スコア: 80%以上が目標
- **シャッフル検出**: 両足が同時に接地し続ける状態を避ける → 無い状態が目標

### 避けるべき歩行パターン（シャッフル）

```
時刻 ─────────────────────────────────────
左足 ─────────────────────...  (常に接地)
     
右足 ─────────────────────...  (常に接地)

両足 xxxxxxxxxxxxxxxxxxxxxx...  (常に接地 = シャッフル！)
```

---

## Walk Score の4つの要素

### 1. Air Time Score (滞空時間スコア) - 重み: 35%

**定義**: 左右の足が地面から離れている時間の割合

**判定基準**:
- 足の高さ > 0.01m → 地面から離れている
- 左足滞空比 + 右足滞空比 → 平均値でスコア化

**スコア計算**:
```
Mean Air Ratio  →  Score
    0%              0.0
   10%              0.1
   30%              0.5
   50%              0.8
   70%              1.0
```

**目標**: 平均滞空比 30-50% で高スコア

### 2. Alternating Gait Score (足の交互性スコア) - 重み: 35%

**定義**: 左足と右足が規則正しく交互に出ているか

**判定方法**:
1. **両足同時接地の割合が小さい** → 交互性が高い
2. **左右の足の切り替えが規則的** → パターンが明確
3. **シャッフル期間がない** → ペナルティなし

**スコア計算**:
```
Alternating Score = 
    Alternating Ratio 
    × (1 - Shuffle Penalty) 
    × (1 - Simultaneous Contact × 0.5)
```

**目標**: 交互性スコア 0.8 以上

### 3. Speed Stability Score (速度安定性スコア) - 重み: 20%

**定義**: 指令速度に対する追従度と速度の安定性

**判定方法**:
1. **速度誤差**: |実速度 - 指令速度| が小さい
2. **速度のばらつき**: 標準偏差が小さい

**スコア計算**:
```
Speed Stability = 
    Tracking Score × 0.6 
    + Smoothness × 0.4
```

**目標**: 0.7 以上

### 4. Symmetry Score (左右対称性スコア) - 重み: 10%

**定義**: 左足と右足の動きが対称的であるか

**判定方法**:
1. **高さの差**: 左右の足の高さの差が小さい
2. **滞空時間のバランス**: 左足と右足の滞空時間がほぼ同じ

**スコア計算**:
```
Symmetry Score = 
    Height Symmetry × 0.6 
    + Air Time Balance × 0.4
```

**目標**: 0.7 以上

---

## 総合 Walk Score

```
Total Walk Score = 
    0.35 × Air Time Score
    + 0.35 × Alternating Score  
    + 0.20 × Speed Stability Score
    + 0.10 × Symmetry Score
```

**スコア範囲**: 0.0 ～ 1.0

---

## スコアの解釈と推奨アクション

### スコア < 0.3: 不良 ❌

**原因**: ロボットが基本的な歩行ができていない（シャッフル）

**推奨対応**:
```python
# リワード調整
reward_scales["feet_air_time"] *= 1.2        # 滞空時間を強く奨励
reward_scales["feet_alternating_pos"] *= 1.1 # 交互性を奨励
reward_scales["tracking_lin_vel"] *= 0.9     # 速度追従は緩和
```

### 0.3 ≤ スコア < 0.6: 中程度 ⚠️

**原因**: 交互パターンが見られるが不安定

**推奨対応**:
```python
# リワード調整
reward_scales["feet_alternating_pos"] *= 1.15 # 交互性をさらに強化
reward_scales["ankle_regularization"] *= 1.1   # 足首姿勢を改善
```

### 0.6 ≤ スコア < 0.8: 良好 ✅

**原因**: 明確な交互パターンが形成されている

**推奨対応**:
```python
# リワード調整
reward_scales["orientation"] *= 1.05       # 体の向きを改善
reward_scales["action_rate"] *= 0.95       # アクション平滑性改善
```

### スコア ≥ 0.8: 優秀 🎉

**原因**: 良好な交互歩行が実現

**推奨対応**:
```python
# リワード調整（微調整）
reward_scales["ankle_regularization"] *= 1.2  # さらに精密な歩行へ
reward_scales["orientation"] *= 1.1
```

---

## メトリクスの詳細 - obs_data から何を見ているか

### 記録されるメトリクス

`eval_metrics.pkl` に保存されるデータ:

```python
metrics = {
    'left_foot_z': [...],      # 左足の高さ (num_steps,)
    'right_foot_z': [...],     # 右足の高さ (num_steps,)
    'lin_vel': [...],          # 前進速度 (num_steps,)
    'command_vel': [...],      # 指令速度 (num_steps,)
    'base_height': [...],      # ベース高さ (num_steps,)
    'base_roll': [...],        # ロール (num_steps,)
    'base_pitch': [...],       # ピッチ (num_steps,)
    'action_magnitude': [...], # アクション大きさ (num_steps,)
    'rewards': [...],          # ステップ報酬 (num_steps,)
}
```

### 足の高さ (left_foot_z, right_foot_z)

`hrp2_env_gs.py` で計算:
```python
def get_link_lowest_point_z(link):
    # リンクのAABB（軸配置境界ボックス）から最下部のZ座標を取得
    # = 足の最低点が地面からどれだけ離れているか
    
l_ankle_z = get_link_lowest_point_z(robot.get_link("LLEG_LINK5"))
r_ankle_z = get_link_lowest_point_z(robot.get_link("RLEG_LINK5"))
```

**判定基準**:
- `z > 0.01m (10mm)` → 地面から離れている（浮遊）
- `z ≤ 0.01m` → 地面に接触している

---

## テスト方法

### 1. 単一の実験のスコアを確認

```bash
python3 test_walk_score.py -e hrp2-walking_loop1 --verbose
```

**出力例**:
```
================================================================================
GAIT PATTERN VISUALIZATION
================================================================================
Pattern (every 5 steps):
XLXLXLXLXLXLXLXLXLXLXL...

Statistics:
  Left foot air time: 40.2%
  Right foot air time: 41.5%
  Both feet in air: 15.3% (両足在宅時間)
  Both feet on ground: 8.2% (両足着地時間 - シャッフル判定基準)

Shuffle Analysis:
  Number of shuffle periods: 2
  Average shuffle length: 3.5 steps

================================================================================
WALK SCORE CALCULATION
================================================================================
📊 Left foot air ratio: 40.20% (height > 0.01m)
📊 Right foot air ratio: 41.50%
📊 Mean air time: 40.85%
📊 Air time score: 0.9843

👣 Simultaneous contact ratio: 8.20% (低いほど良い)
🔄 Alternating pattern ratio: 85.20%
🚶 Shuffle penalty: 2.00%
→ Alternating score: 0.7892

🎯 Velocity tracking error: 0.0234 m/s
📈 Velocity smoothness: 0.8934
⚖️  Tracking score: 0.9234
→ Speed stability: 0.9125

🔄 Left-Right height difference: 0.0042 m
⚖️  Air time balance: 97.50%
→ Symmetry score: 0.9675

────────────────────────────────────────────────────────────────────────────
Walk Quality Metrics for hrp2-walking_loop1:
────────────────────────────────────────────────────────────────────────────
Air Time Score:          0.9843 (足の滞空時間)
Alternating Score:       0.7892 (足の交互性)
Speed Stability Score:   0.9125 (速度安定性)
Symmetry Score:          0.9675 (歩行対称性)
────────────────────────────────────────────────────────────────────────────
TOTAL SCORE:             0.8647
────────────────────────────────────────────────────────────────────────────

Score Interpretation:
  ✅ Good - Clear alternating pattern with minimal shuffling
     Action: Fine-tune for speed and stability
```

### 2. 自動トレーニングループを実行

```bash
python3 hrp2_auto_train_loop.py -e hrp2-walking --num_loops 5
```

ループの各反復で自動的にスコアを計算し、リワードを調整します。

---

## トラブルシューティング

### スコアが常に 0.0

**確認項目**:
1. `eval_metrics.pkl` が正しく保存されているか
   ```bash
   ls -la logs/<exp_name>/eval_metrics.pkl
   ```

2. `hrp2_eval_gs.py` で足の高さが記録されているか
   ```python
   # hrp2_eval_gs.py の出力を確認
   # "Left foot avg height: X.XXX m" という行があるか
   ```

3. 環境の `l_ankle_z` と `r_ankle_z` が正しく計算されているか
   ```python
   # hrp2_env_gs.py の specific_update_buffer() を確認
   ```

### スコアが低い場合

1. **滞空時間スコアが低い**:
   - リワードの `feet_air_time` を増加させる
   - または接地判定基準（0.01m）を調整

2. **交互性スコアが低い**:
   - `feet_alternating_pos` リワードを増加
   - シャッフル期間が多い場合はペナルティを強化

3. **速度安定性が低い**:
   - `tracking_lin_vel` リワードを調整
   - 指令速度を一定に保つ

---

## 実装の詳細

### `hrp2_auto_train_loop.py` の主要メソッド

```python
class RewardOptimizer:
    def calculate_walk_score(self, exp_name):
        """総合歩行スコアを計算"""
        
    def calc_air_time_score(self, metrics):
        """滞空時間スコアを計算"""
        
    def calc_alternating_score(self, metrics):
        """足の交互性スコアを計算"""
        
    def calc_speed_stability(self, metrics):
        """速度安定性スコアを計算"""
        
    def calc_gait_symmetry(self, metrics):
        """歩行対称性スコアを計算"""
        
    def adjust_rewards(self, loop_num, score):
        """スコアに基づいてリワードを調整"""
```

### 判定の流れ

```
eval_policy()
    ↓
各ステップで足の高さを記録
    ↓
eval_metrics.pkl に保存
    ↓
calculate_walk_score()
    ├─ calc_air_time_score() → 滞空時間を判定
    ├─ calc_alternating_score() → 交互性を判定
    ├─ calc_speed_stability() → 速度安定性を判定
    └─ calc_gait_symmetry() → 対称性を判定
    ↓
総合スコア = 加重平均
    ↓
adjust_rewards() → リワード調整
```

---

## 参考: リワード設定例

### 初期リワード（推奨値）

```python
reward_scales = {
    "tracking_lin_vel": 1.0,        # 速度追従
    "tracking_ang_vel": 0.2,        # 角速度追従
    "lin_vel_z": -1.0,              # 上下動ペナルティ
    "base_height": -50.0,           # 転倒ペナルティ
    "action_rate": -0.005,          # アクション平滑化
    "feet_air_time": 2.0,           # ★ 滞空時間（重要）
    "feet_alternating_pos": 3.0,    # ★ 交互性（重要）
    "ankle_regularization": -0.5,   # 足首姿勢
    "orientation": 1.0,             # 体の向き
}
```

### スコア改善のステップ

| Loop | Air Time | Alternating | Speed | Score | 次のアクション |
|------|----------|-------------|-------|-------|----------------|
| 1 | 20% | 40% | 0.5 | 0.35 | Air Time ×1.2 |
| 2 | 35% | 65% | 0.65 | 0.55 | Alternating ×1.15 |
| 3 | 42% | 80% | 0.75 | 0.72 | Fine-tune |
| 4 | 45% | 85% | 0.85 | 0.80 | Minimal adjust |
| 5 | 48% | 87% | 0.90 | 0.82 | ✅ Goal achieved |

---

## 参考資料

- Genesis Simulator: https://genesis-embodied-ai.github.io/
- rsl-rl: Learning locomotion policies
- HRP2 Robot: AIST Humanoid Research Platform

