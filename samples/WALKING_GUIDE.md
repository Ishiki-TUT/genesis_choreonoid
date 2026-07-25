# HRP2 Deterministic Walking Trajectory with Genesis Visualization

C++ の `footstep_planner.cpp` と `stepping_controller.cpp` を Python に移植し、Genesis シミュレータ上でロボットを動かすシステムです。

## ファイル構成

### 歩行計画生成 (Footstep Planning)
- **`footstep_planner.py`** - C++ `footstep_planner.cpp` の Python 移植版
  - `Step`, `Footstep`, `Param`, `Ground` クラス
  - `FootstepPlanner` - 足の配置計画、地面調整、DCM/ZMP 生成
  - 決定論的な歩行軌道生成

- **`hrp2_footplane.py`** - 歩行計画の実行スクリプト
  - 5ステップの歩行計画を生成
  - JSON ファイルで軌跡を保存

### 歩行制御 (Walking Control)
- **`stepping_controller.py`** - C++ `stepping_controller.cpp` の Python 移植版
  - ステップタイミング制御
  - 足の軌跡生成

### Genesis シミュレーション
- **`hrp2_walking_sim.py`** - 基本的な歩行シミュレーション
  - Genesis 上でロボットを動かす
  - 簡単な関節制御

- **`hrp2_walking_sim_ik.py`** - 逆運動学を使った高度なシミュレーション
  - 簡易的な IK ソルバー実装
  - より精密な脚の軌跡制御

- **`hrp2_walking_visualizer.py`** - 推奨：既存RL環境との統合版
  - 既存の RL 環境を活用
  - 最も安定した実行環境

## 使い方

### 1. 歩行計画を生成・確認

```bash
# 標準パラメータで5ステップの計画を生成
python3 hrp2_footplane.py --verbose

# カスタムパラメータで生成
python3 hrp2_footplane.py \
    --stride 0.25 \
    --spacing 0.18 \
    --com-height 0.75 \
    --step-duration 0.6 \
    --verbose
```

オプション:
- `--stride`: 前進距離 (m) [デフォルト: 0.2]
- `--sway`: 横揺れ (m) [デフォルト: 0.0]
- `--turn`: 回転角度 (rad) [デフォルト: 0.0]
- `--spacing`: 足の間隔 (m) [デフォルト: 0.2]
- `--com-height`: CoM の高さ (m) [デフォルト: 0.8]
- `--T`: 時定数 (s) [デフォルト: 1.0]
- `--step-duration`: 各ステップの時間 (s) [デフォルト: 0.8]
- `--output`: 出力ファイル名 [デフォルト: hrp2_5step_trajectory.json]
- `--verbose`: 詳細情報を表示

出力例:
```
======================================================================
HRP2 Deterministic Walking Trajectory Generator
======================================================================

Parameters:
  Stride: 0.2000 m
  Sway: 0.0000 m
  Turn: 0.0000 rad
  Spacing: 0.2000 m
  CoM height: 0.8000 m
  Time constant (T): 1.0000 s
  Step duration: 0.8000 s

Generating 5-step walking plan...
✓ Successfully generated 5 steps

Trajectory Summary
======================================================================
Step  0: Support: LEFT  | Swing displacement: 0.4472 m
Step  1: Support: RIGHT | Swing displacement: 0.4000 m
Step  2: Support: LEFT  | Swing displacement: 0.4000 m
Step  3: Support: RIGHT | Swing displacement: 0.4000 m

Total forward distance: 1.6472 m
Total time: 4.0000 s
Average velocity: 0.4118 m/s
```

### 2. Genesis で可視化・シミュレーション実行

#### 方法 A: 推奨（既存RL環境統合版）

```bash
# 標準パラメータで実行
python3 hrp2_walking_visualizer.py --duration 10

# ビューアーなしで実行（高速）
python3 hrp2_walking_visualizer.py --duration 20 --no-viewer

# 軌跡を保存
python3 hrp2_walking_visualizer.py \
    --duration 10 \
    --save-trajectory walking_data.json

# CPU で実行
python3 hrp2_walking_visualizer.py --device cpu --duration 5
```

オプション:
- `--duration`: シミュレーション時間 (s) [デフォルト: 10]
- `--dt`: シミュレーション時間ステップ (s) [デフォルト: 0.01]
- `--num-envs`: 並列環境数 [デフォルト: 1]
- `--no-viewer`: ビューアーを非表示
- `--device`: 計算デバイス (cuda/cpu) [デフォルト: cuda]
- `--save-trajectory`: 軌跡を保存するファイル名
- `--quiet`: 詳細出力を非表示

#### 方法 B: 逆運動学版（高度な制御）

```bash
python3 hrp2_walking_sim_ik.py --duration 10

# IK を無効化してシンプルに
python3 hrp2_walking_sim_ik.py --no-ik --duration 5
```

#### 方法 C: 基本版

```bash
python3 hrp2_walking_sim.py --duration 10
```

## パラメータ調整ガイド

### 歩行パターンの変更

```bash
# ゆっくり歩く
python3 hrp2_footplane.py \
    --stride 0.15 \
    --step-duration 1.0 \
    --verbose

# 速く歩く
python3 hrp2_footplane.py \
    --stride 0.25 \
    --step-duration 0.6 \
    --verbose

# 横向き歩行
python3 hrp2_footplane.py \
    --stride 0.0 \
    --sway 0.2 \
    --verbose

# 回転歩行
python3 hrp2_footplane.py \
    --turn 0.1 \
    --verbose
```

### COM高さの調整

```bash
# 低い姿勢
python3 hrp2_footplane.py --com-height 0.70 --verbose

# 高い姿勢
python3 hrp2_footplane.py --com-height 0.90 --verbose
```

## 出力ファイル

### JSON フォーマット

`hrp2_5step_trajectory.json` の構造:

```json
{
  "timestamp": "2026-05-01T...",
  "parameters": {
    "com_height": 0.8,
    "T": 1.0
  },
  "steps": [
    {
      "index": 0,
      "support_foot": 0,
      "left_foot": {
        "position": [-0.1, 0.1, 0.0],
        "angle_rpy": [0.0, 0.0, 0.0]
      },
      "right_foot": {
        "position": [-0.1, -0.1, 0.0],
        "angle_rpy": [0.0, 0.0, 0.0]
      },
      "zmp": [-0.1, 0.0, 0.0],
      "dcm": [0.0, 0.0, 0.8],
      "duration": 0.8,
      "stepping": true
    },
    ...
  ]
}
```

## 技術詳細

### 歩行計画アルゴリズム

1. **足の配置計画** (`plan()`)
   - サポート足と スウィング足を交互に配置
   - 各ステップパラメータ（stride, sway, turn）を適用
   - 相対位置を計算し、ワールド座標に変換

2. **地面調整** (`align_to_ground()`)
   - 足を地面に接地させる
   - 地面の傾きに合わせて足の姿勢を調整

3. **DCM/ZMP 生成** (`generate_dcm()`)
   - 逆動力学を使用して DCM (Divergent Component of Motion) を計算
   - Zero Moment Point (ZMP) を決定
   - ステップの安定性を確保

### 主要な物理パラメータ

- **CoM 高さ** (`com_height`): 質量中心の高さ
  - 低い → 安定だが動きが制限される
  - 高い → 動きやすいが安定性が低下

- **時定数** (`T`): 動的特性の時定数
  - 大きい → 緩やかな動き
  - 小さい → 急速な応答

- **ステップ時間** (`duration`): 各ステップの継続時間
  - 長い → ゆっくり歩く
  - 短い → 速く歩く

## トラブルシューティング

### Genesis が見つからない

```
ImportError: No module named 'genesis'
```

**解決方法:**
```bash
# Genesis をインストール
pip install genesis-sim
```

### ビューアーが表示されない

- CUDA 対応 GPU がある場合は自動的に GPU 計算
- `--device cpu` で CPU モードに切り替え
- `--no-viewer` でビューアーなしで実行

### URDF ファイルが見つからない

`hrp2.urdf` が見つからない場合、代替 URDF が自動的に探索されます。

### メモリ不足エラー

```bash
# 環境数を減らす
python3 hrp2_walking_visualizer.py --num-envs 1

# CPU で実行
python3 hrp2_walking_visualizer.py --device cpu
```

## 次のステップ

1. **より複雑な軌跡生成**
   - 階段登り（`climb` パラメータ）
   - より細かいステップ制御

2. **実ロボットへの統合**
   - Choreonoid シミュレータとの連携
   - 実ハードウェアへのデプロイ

3. **強化学習との結合**
   - 生成された軌跡をベースとした学習
   - ポリシーの調整

## 参考資料

- C++ オリジナル: `footstep_planner.cpp`, `stepping_controller.cpp`
- Footstep Planner 理論: 足の順運動学・逆運動学、DCM/ZMP
- DCM 理論: Kajita et al. "Biped Walking Pattern Generation by using Linear Inverted Pendulum Mode"

## ライセンス

このコードは IRSL による実装です。オリジナルの C++ コードから移植されました。
