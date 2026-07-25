# HRP2 Walking Simulation - Genesis公式API準拠への移行

## 概要

`hrp2_walking_sim.py`を**Genesis 0.4.7公式API仕様に完全準拠**するように改修しました。

## 主な変更点

### 1. 関節インデックスの取得方法

**変更前:**
```python
self.dof_names = self.robot.dof_names()  # 非推奨な方法
```

**変更後（公式推奨）:**
```python
self.joint_names = ["LLEG_JOINT0", "LLEG_JOINT1", ...]
self.dofs_idx_local = []
for joint_name in self.joint_names:
    dof_idx = self.robot.get_joint(joint_name).dof_idx_local
    self.dofs_idx_local.append(dof_idx)
self.dofs_idx_local = np.array(self.dofs_idx_local)
```

**理由:**
- Genesis公式ドキュメントで推奨されている方法
- `dof_idx_local`: ロボットエンティティ内でのローカルなDOFインデックス
- `dof_idx`: シーン内での全局的なDOFインデックス

### 2. PD制御ゲイン設定

**変更前:**
```python
kp = torch.full((n_dofs,), 300.0, dtype=torch.float32, device='cuda')
kv = torch.full((n_dofs,), 15.0, dtype=torch.float32, device='cuda')
self.robot.set_dofs_kp(kp.cpu().numpy())
self.robot.set_dofs_kv(kv.cpu().numpy())
```

**変更後（公式推奨）:**
```python
kp_values = np.array([500.0, 500.0, 300.0, ...], dtype=np.float32)
kv_values = np.array([20.0, 20.0, 10.0, ...], dtype=np.float32)
self.robot.set_dofs_kp(kp_values, self.dofs_idx_local)
self.robot.set_dofs_kv(kv_values, self.dofs_idx_local)
self.robot.set_dofs_force_range(force_lower, force_upper, self.dofs_idx_local)
```

**理由:**
- `dofs_idx_local`パラメータで対象DOFを明示的に指定
- 安全な力の範囲を設定して、物理的な妥当性を確保

### 3. 初期姿勢設定

**変更前:**
```python
for i, angle in enumerate(initial_angles):
    target = torch.tensor([angle], dtype=torch.float32, device='cuda')
    self.robot.set_dof_target(target, i)
```

**変更後（公式推奨）:**
```python
# 硬いリセット: physics を無視して直接位置を設定
initial_angles = np.array([...], dtype=np.float32)
for step in range(50):
    self.robot.set_dofs_position(initial_angles, self.dofs_idx_local)
    self.scene.step()
```

**理由:**
- `set_dofs_position`: 複数DOFを一括設定でき、効率的
- `dofs_idx_local`で対象DOFを指定

### 4. 関節角度制御 - 単一関節

**変更前:**
```python
def set_joint_target(self, joint_name, angle):
    target = torch.tensor([angle], dtype=torch.float32, device='cuda')
    self.robot.set_dof_target(target, idx)
```

**変更後（公式推奨）:**
```python
def set_joint_target(self, joint_name, angle):
    target_angles = np.array([angle], dtype=np.float32)
    dof_indices = np.array([dof_idx], dtype=np.int32)
    self.robot.control_dofs_position(target_angles, dof_indices)
```

### 5. 関節角度制御 - 全関節の一括設定

**新規追加:**
```python
def set_all_joint_targets(self, angles):
    """すべての制御対象関節の目標角度を一括設定"""
    angles = np.asarray(angles, dtype=np.float32)
    self.robot.control_dofs_position(angles, self.dofs_idx_local)
```

**重要:** Genesis公式API仕様では、`control_dofs_position`で一度設定した目標値は保持され続けるため、毎フレーム設定し直す必要はない。

### 6. 歩行軌道生成

**改善:**
```python
def _update_joint_targets_from_feet(self):
    # 足の位置・姿勢から関節角度を計算
    target_angles = np.array([
        0.0, left_hip_pitch, 0.0, left_knee, left_ankle_pitch, 0.0,  # Left
        0.0, right_hip_pitch, 0.0, right_knee, right_ankle_pitch, 0.0,  # Right
    ], dtype=np.float32)
    # 一括設定
    self.set_all_joint_targets(target_angles)
```

## Genesis公式API仕様の主要メソッド

### 1. 関節情報取得
```python
joint = robot.get_joint(joint_name)
dof_idx_local = joint.dof_idx_local    # ローカルDOFインデックス
dof_idx_global = joint.dof_idx         # グローバルDOFインデックス
```

### 2. PD制御ゲイン設定
```python
robot.set_dofs_kp(kp_array, dofs_idx_local)      # 位置ゲイン
robot.set_dofs_kv(kv_array, dofs_idx_local)      # 速度ゲイン
robot.set_dofs_force_range(lower, upper, dofs_idx_local)  # 力制限
```

### 3. 位置制御（PD制御）
```python
# 一度設定すると保持される
robot.control_dofs_position(target_positions, dofs_idx_local)
robot.control_dofs_velocity(target_velocities, dofs_idx_local)
robot.control_dofs_force(target_forces, dofs_idx_local)
```

### 4. 硬いリセット（physics無視）
```python
robot.set_dofs_position(positions, dofs_idx_local)
```

### 5. 状態取得
```python
positions = robot.get_dofs_position(dofs_idx_local)
velocities = robot.get_dofs_velocity(dofs_idx_local)
forces = robot.get_dofs_force(dofs_idx_local)
control_forces = robot.get_dofs_control_force(dofs_idx_local)
```

## 動作確認

```bash
# デフォルト設定で実行
python samples/hrp2_walking_sim.py

# カスタムパラメータで実行
python samples/hrp2_walking_sim.py --stride 0.3 --com-height 0.85 --duration 20.0
```

## 参考

- Genesis公式ドキュメント: https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html
- `rl_env_gs.py`の実装も参考にしました

## 今後の改善予定

1. **逆運動学の精密化**: 現在は簡略版（高さベース）なので、完全な3D IKの実装
2. **歩行安定性の向上**: より詳細な足の軌道計画（サイクロイド軌道など）
3. **シミュレーション精度向上**: 接地判定、摩擦制御の改善
4. **パラレルシミュレーション**: 複数環境での並列実行対応
