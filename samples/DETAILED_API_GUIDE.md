# HRP2 Walking Simulation - API改修の詳細解説

## 🏗️ アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────┐
│         HRP2 Walking Simulation System                   │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    ┌────────┐    ┌────────┐    ┌──────────┐
    │Genesis │    │Footstep│    │Stepping  │
    │Scene   │    │Planner │    │Controller│
    └────────┘    └────────┘    └──────────┘
        │               │               │
        │               └───────┬───────┘
        │                       │
        ▼                       ▼
    ┌────────────────────────────────────┐
    │  HRP2WalkingSimulator              │
    ├────────────────────────────────────┤
    │ _init_genesis()                    │
    │ _setup_robot_info()               │ ← Genesis API改修
    │ _setup_pd_control()               │ ← Genesis API改修
    │ _set_initial_pose()               │ ← Genesis API改修
    │ set_joint_target()                │ ← Genesis API改修
    │ set_all_joint_targets()           │ ← 新規追加
    │ _update_joint_targets_from_feet() │ ← Genesis API改修
    │ step_simulation()                 │
    │ run()                             │
    └────────────────────────────────────┘
```

---

## 🔄 制御フロー

### 初期化フロー
```
HRP2WalkingSimulator.__init__()
    ↓
_init_genesis()
    ├─ gs.init(backend=gs.cuda)
    ├─ scene = gs.Scene(...)
    ├─ robot = scene.add_entity(URDF)
    ├─ scene.build()
    ├─ _setup_robot_info()     ← ① 関節インデックス取得
    ├─ _setup_pd_control()     ← ② PDゲイン設定
    └─ _set_initial_pose()     ← ③ 初期姿勢設定

generate_walking_plan()
    ├─ footstep, param = generate_5step_walking_plan(...)
    └─ _setup_stepping_controller()
```

### シミュレーション実行フロー
```
for t in range(num_steps):
    step_simulation()
        ├─ stepping_controller.update(...)
        │   ↓
        │   足の軌道を生成
        │   (pos_ref, angle_ref)
        │
        ├─ _update_joint_targets_from_feet()
        │   ├─ 足の位置から関節角度を計算
        │   ├─ set_all_joint_targets(angles)
        │   │   ↓
        │   │   robot.control_dofs_position(angles, dofs_idx_local)
        │   │   ← ④ 公式API: 一括PD制御
        │   └─（以下、自動で保持される）
        │
        └─ scene.step()
            ↓
            Genesis物理シミュレータが実行
            （自動でPD制御則を計算・適用）
```

---

## 🔑 Key改修 - Genesis公式API準拠

### ① 関節インデックス取得

```python
# ❌ 旧方法（非推奨）
dof_names = robot.dof_names()
idx = dof_names.index("LLEG_JOINT0")

# ✓ 新方法（推奨）
dof_idx_local = robot.get_joint("LLEG_JOINT0").dof_idx_local
```

**変更理由:**
- `dof_idx_local`: ロボット内でのローカルインデックス（複数環境でも一貫性あり）
- `dof_idx`: グローバルインデックス（シーン全体での一意なID）
- 公式ドキュメントで推奨

**データ構造:**
```python
self.joint_names = [
    "LLEG_JOINT0", "LLEG_JOINT1", ..., "RLEG_JOINT5"
]
self.dofs_idx_local = np.array([0, 1, ..., 11])  # ローカルインデックス
self.num_motors = 12  # HRP2の脚の自由度
```

---

### ② PDゲイン設定

**旧コード:**
```python
kp = torch.full((n_dofs,), 300.0, dtype=torch.float32, device='cuda')
robot.set_dofs_kp(kp.cpu().numpy())
```

**新コード:**
```python
kp_values = np.array([
    500.0, 500.0, 300.0, 500.0, 300.0, 300.0,  # Left leg
    500.0, 500.0, 300.0, 500.0, 300.0, 300.0,  # Right leg
], dtype=np.float32)

kv_values = np.array([
    20.0, 20.0, 10.0, 20.0, 10.0, 10.0,
    20.0, 20.0, 10.0, 20.0, 10.0, 10.0,
], dtype=np.float32)

robot.set_dofs_kp(kp_values, self.dofs_idx_local)
robot.set_dofs_kv(kv_values, self.dofs_idx_local)
robot.set_dofs_force_range(force_lower, force_upper, self.dofs_idx_local)
```

**改善点:**
- ✓ `dofs_idx_local`で対象を明示的に指定
- ✓ 関節ごとのゲイン設定が可能
- ✓ 力の安全範囲を設定可能
- ✓ 複数環境でも正確に適用

---

### ③ 初期姿勢設定

**旧コード:**
```python
for i, angle in enumerate(initial_angles):
    target = torch.tensor([angle], dtype=torch.float32, device='cuda')
    robot.set_dof_target(target, i)  # ❌ 廃止API
```

**新コード:**
```python
# Hard reset (physics無視): 複数フレームで段階的に設定
for step in range(50):
    robot.set_dofs_position(initial_angles, self.dofs_idx_local)
    scene.step()

# Settling: 落ち着かせる
for _ in range(100):
    scene.step()
```

**改善点:**
- ✓ `set_dofs_position()`: 複数DOF効率的に設定
- ✓ 段階的設定でシミュレーション安定性向上
- ✓ `dofs_idx_local`で対象明示

---

### ④ 関節角度制御 - 公式API

**旧コード:**
```python
# ❌ 廃止されたAPI
robot.set_dof_target(target, i)
```

**新コード:**
```python
# ✓ 推奨: control_dofs_position（PD制御）
target_angles = np.array([0.0, 0.2, 0.3, ...], dtype=np.float32)
robot.control_dofs_position(target_angles, self.dofs_idx_local)

# または力制御
robot.control_dofs_force(force_commands, self.dofs_idx_local)

# または速度制御
robot.control_dofs_velocity(velocity_commands, self.dofs_idx_local)
```

**重要な特性:**
```
once control_dofs_position is set, 
it will be stored internally and you don't need to send 
repetitive commands to the simulation in the following steps 
as long as your target stays the same.
```
→ つまり、毎フレーム設定し直す必要はない！

---

## 📊 シミュレーション実行の流れ（図解）

```
Step 0: 初期化
┌─────────────────────────────────┐
│ _setup_robot_info()             │
│  └─ dofs_idx_local = [0,1,...]  │
│ _setup_pd_control()             │
│  └─ set_dofs_kp/kv()            │
│ _set_initial_pose()             │
│  └─ set_dofs_position()         │
└─────────────────────────────────┘
            ↓
Step 1-500: 歩行実行
┌─────────────────────────────────┐
│ For each simulation step:        │
│  1. stepping_controller.update() │
│     → 足の軌道生成              │
│  2. _update_joint_targets()     │
│     → 関節角度計算              │
│     → control_dofs_position()   │ ← ★ 公式API
│       （設定値は自動で保持される）  │
│  3. scene.step()                │
│     → Genesis物理シミュレータ    │
│     → PD制御則を自動計算・適用  │
│     → 関節トルク発生            │
│     → 動力学計算                │
│     → ロボット状態更新          │
└─────────────────────────────────┘
            ↓
Step 501: 終了
```

---

## 📈 パフォーマンス最適化

### メモリ使用量削減
```python
# 旧: 毎フレーム新しいテンソル生成
for i in range(num_steps):
    target = torch.tensor([angle], dtype=torch.float32, device='cuda')  # ← メモリ確保
    robot.set_dof_target(target, i)

# 新: NumPy配列（CPU）で効率的
target_angles = np.array([...], dtype=np.float32)  # 一度だけ割り当て
for i in range(num_steps):
    robot.control_dofs_position(target_angles, dofs_idx_local)  # 同じ配列再利用
```

### 通信オーバーヘッド削減
```python
# ✓ 新方法: 一度設定すると保持される
robot.control_dofs_position(targets, dofs_idx_local)
for _ in range(1000):
    scene.step()  # targets は自動で保持される

# ❌ 旧方法: 毎フレーム通信
for _ in range(1000):
    robot.set_dof_target(target, i)  # ← 毎フレーム通信
    scene.step()
```

---

## 🧪 検証チェックリスト

- [ ] コンパイルエラーなし: `python -m py_compile hrp2_walking_sim.py`
- [ ] Genesis API テスト実行: `python test_genesis_api.py`
- [ ] HRP2 URDF ロード成功
- [ ] 関節インデックス取得成功（12個）
- [ ] PD制御ゲイン設定成功
- [ ] 初期姿勢設定成功（床めり込みなし）
- [ ] シミュレーション実行（クラッシュなし）
- [ ] 歩行軌道が滑らか
- [ ] 足が正しく上がる・下りる

---

## 🔗 参考リンク

1. **Genesis 公式ドキュメント**
   - Control: https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html
   - API Ref: https://genesis-world.readthedocs.io/en/latest/api/entities/robots.html

2. **実装参考**
   - `irsl_rl/rl_env_gs.py`: 公式API準拠の参考実装
   - `samples/bex24_env_gs.py`: 他ロボットの例
   - `samples/hrp2_eval_gs.py`: HRP2評価スクリプト
