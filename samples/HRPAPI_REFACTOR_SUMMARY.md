# HRP2 Walking Simulation - Genesis API改修サマリー

## 🔧 改修内容

Genesis 0.4.7公式ドキュメントに完全準拠するように、`hrp2_walking_sim.py`を改修しました。

### 改修ファイル
- `/home/irsl/Documents/genesis_choreonoid/samples/hrp2_walking_sim.py`

## 📋 変更内容の詳細

### 1️⃣ 関節インデックス取得 (`_setup_robot_info`)

**公式推奨パターン適用:**
```python
# ✓ 改修後（推奨）
for joint_name in self.joint_names:
    dof_idx = self.robot.get_joint(joint_name).dof_idx_local
    self.dofs_idx_local.append(dof_idx)
self.dofs_idx_local = np.array(self.dofs_idx_local)
```

**利点:**
- ✓ `dof_idx_local`を明示的に取得
- ✓ ロボットの仕様変更に強い
- ✓ 公式ドキュメント推奨方法

---

### 2️⃣ PD制御ゲイン設定 (`_setup_pd_control`)

**改修前:**
```python
kp = torch.full((n_dofs,), 300.0, dtype=torch.float32, device='cuda')
self.robot.set_dofs_kp(kp.cpu().numpy())
```

**改修後:**
```python
kp_values = np.array([500.0, 500.0, ...], dtype=np.float32)
kv_values = np.array([20.0, 20.0, ...], dtype=np.float32)
self.robot.set_dofs_kp(kp_values, self.dofs_idx_local)
self.robot.set_dofs_kv(kv_values, self.dofs_idx_local)
self.robot.set_dofs_force_range(force_lower, force_upper, self.dofs_idx_local)
```

**利点:**
- ✓ `dofs_idx_local`で対象DOFを明示的に指定
- ✓ 力の範囲制限で物理的な安全性を確保
- ✓ より細かいゲイン調整が可能

---

### 3️⃣ 初期姿勢設定 (`_set_initial_pose`)

**改修前:**
```python
for i, angle in enumerate(initial_angles):
    target = torch.tensor([angle], dtype=torch.float32, device='cuda')
    self.robot.set_dof_target(target, i)  # ❌ 廃止API
```

**改修後:**
```python
# 硬いリセット: physics を無視して直接設定
for step in range(50):
    self.robot.set_dofs_position(initial_angles, self.dofs_idx_local)
    self.scene.step()
```

**利点:**
- ✓ `set_dofs_position`で複数DOFを効率的に設定
- ✓ `dofs_idx_local`で対象DOFを指定
- ✓ 複数フレームで段階的に設定し、シミュレーション安定性向上

---

### 4️⃣ 関節角度制御 - 単一関節 (`set_joint_target`)

**改修前:**
```python
target = torch.tensor([angle], dtype=torch.float32, device='cuda')
self.robot.set_dof_target(target, idx)  # ❌ 廃止API
```

**改修後:**
```python
target_angles = np.array([angle], dtype=np.float32)
dof_indices = np.array([dof_idx], dtype=np.int32)
self.robot.control_dofs_position(target_angles, dof_indices)  # ✓ PD制御
```

---

### 5️⃣ 関節角度制御 - 一括設定（新規追加）

**新規メソッド:**
```python
def set_all_joint_targets(self, angles):
    """すべての制御対象関節の目標角度を一括設定"""
    angles = np.asarray(angles, dtype=np.float32)
    self.robot.control_dofs_position(angles, self.dofs_idx_local)
```

**利点:**
- ✓ 複数DOFを一度に制御
- ✓ `control_dofs_position`の設定値は自動で保持される（毎フレーム設定不要）
- ✓ 性能向上

---

### 6️⃣ 歩行軌道更新 (`_update_joint_targets_from_feet`)

**改修後:**
```python
# 足位置から関節角度を計算
target_angles = np.array([
    0.0, left_hip_pitch, 0.0, left_knee, left_ankle_pitch, 0.0,      # Left leg
    0.0, right_hip_pitch, 0.0, right_knee, right_ankle_pitch, 0.0,    # Right leg
], dtype=np.float32)

# 一括制御で効率化
self.set_all_joint_targets(target_angles)
```

**利点:**
- ✓ シミュレーションループの効率化
- ✓ 全脚関節の同期制御

---

## 🎯 Genesis公式API仕様との対応

| 用途 | 推奨API | 旧API |
|------|--------|-------|
| **関節情報取得** | `robot.get_joint(name).dof_idx_local` | `robot.dof_names()` |
| **PD制御設定** | `set_dofs_kp(kp, dofs_idx_local)` | なし |
| **速度ゲイン設定** | `set_dofs_kv(kv, dofs_idx_local)` | なし |
| **力範囲設定** | `set_dofs_force_range(lower, upper, dofs_idx_local)` | なし |
| **位置制御** | `control_dofs_position(targets, dofs_idx_local)` | `set_dof_target()` ❌ |
| **硬いリセット** | `set_dofs_position(positions, dofs_idx_local)` | `set_dof_target()` ❌ |
| **状態取得** | `get_dofs_position(dofs_idx_local)` | なし |

---

## ✅ テスト方法

### 1. コンパイルチェック
```bash
python -m py_compile samples/hrp2_walking_sim.py
```

### 2. Genesis API テスト
```bash
python samples/test_genesis_api.py
```

### 3. 歩行シミュレーション実行
```bash
# デフォルト設定
python samples/hrp2_walking_sim.py

# カスタム設定
python samples/hrp2_walking_sim.py \
  --stride 0.3 \
  --com-height 0.85 \
  --duration 20.0 \
  --dt 0.01
```

---

## 📚 参考資料

### Genesis公式ドキュメント
- **Control Your Robot**: https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html
- **API Reference**: https://genesis-world.readthedocs.io/en/latest/api/index.html

### 関連ファイル
- `irsl_rl/rl_env_gs.py` - 公式API準拠の参考実装
- `samples/hrp2_eval_gs.py` - HRP2評価スクリプト
- `samples/bex24_env_gs.py` - 他のロボット環境実装例

---

## 🚀 今後の予定

1. **逆運動学の精密化**
   - 現在: 高さベースの簡略版
   - 今後: 完全な3D IK実装

2. **歩行安定性向上**
   - ZMP制御の実装
   - 接地判定の改善
   - サイクロイド軌道の採用

3. **マルチエンビロンメント対応**
   - Genesis のパラレルシミュレーション機能対応

4. **物理シミュレーション精度向上**
   - 摩擦パラメータの最適化
   - 接触判定の改善
   - アクチュエータダイナミクスの実装

---

## 📝 注記

- Genesis 0.4.7以上での動作を想定しています
- CUDA対応GPUを使用していますが、CPUモード(`backend=gs.cpu`)でも動作します
- URDFファイル: `hrp2_description/HRP2_genesis.urdf`
