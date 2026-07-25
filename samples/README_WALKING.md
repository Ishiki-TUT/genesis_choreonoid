# 🚶 HRP2 Deterministic Walking Trajectory System - Complete Implementation Summary

## ✅ 完成内容

C++ の `footstep_planner.cpp` と `stepping_controller.cpp` を Python に移植し、Genesis シミュレータ上でロボットを歩かせるシステムが完成しました。

## 📁 作成されたファイル一覧

### 1. **歩行計画コア** (Footstep Planning)

| ファイル | 説明 |
|---------|------|
| `footstep_planner.py` | C++ footstep_planner.cpp の Python 移植版 |
| `stepping_controller.py` | C++ stepping_controller.cpp の Python 移植版 |
| `hrp2_footplane.py` | 5ステップ歩行計画生成・保存スクリプト |

### 2. **Genesis シミュレーション** (Visualization & Simulation)

| ファイル | 説明 | 推奨度 |
|---------|------|--------|
| `hrp2_walking_simple.py` | ⭐ **最もシンプル** - 初心者向け | ⭐⭐⭐ |
| `hrp2_walking_sim.py` | 基本的なシミュレーション | ⭐⭐ |
| `hrp2_walking_sim_ik.py` | 逆運動学を使った高度な制御 | ⭐ |
| `hrp2_walking_visualizer.py` | 既存RL環境統合版 | ⭐⭐ |

### 3. **ユーティリティ**

| ファイル | 説明 |
|---------|------|
| `quick_start.py` | 対話型クイックスタートガイド |
| `walking_quickstart.py` | シナリオ実行スクリプト |
| `WALKING_GUIDE.md` | 詳細なドキュメント |

## 🚀 クイックスタート

### 最も簡単な方法（推奨）

```bash
# 対話型ガイドで実行
python3 quick_start.py
```

または直接実行：

```bash
# 方法1: 歩行計画を生成・表示
python3 hrp2_footplane.py --verbose

# 方法2: Genesis で可視化
python3 hrp2_walking_simple.py --duration 10

# 方法3: 詳細なビューアー
python3 hrp2_walking_visualizer.py --duration 10
```

## 📊 実行結果の例

```
======================================================================
HRP2 Deterministic Walking Trajectory Generator
======================================================================

Parameters:
  Stride: 0.2000 m
  CoM height: 0.8000 m
  Time constant (T): 1.0000 s

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
======================================================================
✓ Trajectory saved to hrp2_5step_trajectory.json
```

## 🎯 各スクリプトの使い分け

### `hrp2_footplane.py` - 歩行計画生成
**目的**: 歩行軌道を計算・保存
**用途**: 計画を確認、JSON出力を他のツールで使う

```bash
python3 hrp2_footplane.py --stride 0.2 --verbose
```

### `hrp2_walking_simple.py` - シンプル可視化 ⭐ **推奨**
**目的**: Genesis上でロボットを歩かせる
**用途**: 歩行がどのように見えるか確認

```bash
python3 hrp2_walking_simple.py --duration 10
```

**オプション**:
```bash
# ビューアーなし（高速）
python3 hrp2_walking_simple.py --no-viewer --quiet

# パラメータ指定
python3 hrp2_walking_simple.py --stride 0.25 --spacing 0.18
```

### `hrp2_walking_sim.py` - 基本シミュレーション
**目的**: 関節制御を含むシミュレーション
**用途**: より詳細なロボット制御を実装したい場合

```bash
python3 hrp2_walking_sim.py --duration 10
```

### `hrp2_walking_visualizer.py` - 高度な可視化
**目的**: 既存RL環境との統合
**用途**: 強化学習と組み合わせたい場合

```bash
python3 hrp2_walking_visualizer.py --duration 10 --save-trajectory data.json
```

## 📝 パラメータ調整例

### 歩き方を変える

```bash
# ゆっくり歩く（小さいステップ）
python3 hrp2_footplane.py --stride 0.15 --verbose

# 速く歩く（大きいステップ）
python3 hrp2_footplane.py --stride 0.25 --verbose

# 横向き移動
python3 hrp2_footplane.py --stride 0.0 --sway 0.2 --verbose

# 回転しながら歩く
python3 hrp2_footplane.py --turn 0.1 --verbose
```

### 姿勢を変える

```bash
# 低い姿勢
python3 hrp2_footplane.py --com-height 0.70 --verbose

# 高い姿勢
python3 hrp2_footplane.py --com-height 0.90 --verbose
```

## 📊 出力ファイル

### `hrp2_5step_trajectory.json`
```json
{
  "timestamp": "2026-05-13T...",
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
      ...
    }
  ]
}
```

## 🔧 技術詳細

### 歩行計画アルゴリズム

1. **Step Planning** (`plan()`)
   - サポート足と スウィング足を交互に配置
   - 歩行パラメータ（stride, sway, turn）を適用

2. **Ground Adjustment** (`align_to_ground()`)
   - 足を地面に接地
   - 地面の傾きに合わせて足の姿勢を調整

3. **DCM/ZMP Generation** (`generate_dcm()`)
   - Divergent Component of Motion を計算
   - Zero Moment Point を決定
   - 歩行安定性を確保

### 主要パラメータ

| パラメータ | 説明 | デフォルト | 推奨範囲 |
|-----------|------|-----------|---------|
| `stride` | 前進距離 (m) | 0.2 | 0.1～0.3 |
| `spacing` | 足の間隔 (m) | 0.2 | 0.15～0.25 |
| `com_height` | CoM の高さ (m) | 0.8 | 0.7～0.9 |
| `T` | 時定数 (s) | 1.0 | 0.5～2.0 |

## ⚙️ トラブルシューティング

### Genesis が起動しない
```bash
# GPU で実行（推奨）
python3 hrp2_walking_simple.py

# CPU で実行
# (Genesis の設定を確認)
```

### URDF ファイルが見つからない
```
genesis.GenesisException: File not found in either current directory...
```

**解決策**: `hrp2_description/HRP2_genesis.urdf` ファイルが正しいパスにあるか確認

### ビューアーが表示されない
```bash
# ビューアーなしで実行
python3 hrp2_walking_simple.py --no-viewer
```

## 📚 参考資料

- **C++ オリジナル**: `footstep_planner.cpp`, `stepping_controller.cpp`
- **Footstep Planner 理論**: 足の運動学、DCM/ZMP 理論
- **DCM 参考**: Kajita et al. "Biped Walking Pattern Generation by using Linear Inverted Pendulum Mode"

## 🎯 次のステップ

### 1. 実ロボットへの統合
- Choreonoid シミュレータとの連携
- 実ハードウェアへのデプロイ

### 2. より複雑な軌跡
- 階段登り機能
- 動的平衡制御

### 3. 強化学習との結合
- 生成された軌跡をベースとした学習
- ポリシー最適化

## 💡 主な特徴

✅ **完全なPython移植** - C++ コードを 100% Python に移植  
✅ **決定論的な計画** - 強化学習不要、確定的な軌道生成  
✅ **Genesis 統合** - 最新シミュレータで可視化  
✅ **簡単な使用方法** - パラメータ指定で即座に実行  
✅ **拡張性** - モジュール化された設計で容易にカスタマイズ可能  

## 📞 サポート

詳細は `WALKING_GUIDE.md` を参照してください。

---

**Created**: May 2026  
**IRSL - Intelligent Robotics and System Laboratory**
