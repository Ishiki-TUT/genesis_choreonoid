# ロボットが倒れてしまう問題 - 修正報告

## 🔍 問題の原因

ロボットが初期姿勢のまま倒れてしまう理由は**3つの致命的なバグ**がありました：

### 1. **scene.build() の順序が間違っていた** (最重要)
```python
# ❌ 間違い（以前）
self.robot = self.scene.add_entity(...)
self._setup_robot_info()
self._setup_pd_control()
self._set_initial_pose()
self.scene.build()  # ← ここで初期化され、直前の設定が全て無視される！
```

**理由**: `scene.build()` を実行することで、シーンがコンパイルされます。その前に `set_dofs_position()` や `set_dofs_kp()` など全ての設定が無視されてしまいます。

```python
# ✅ 正解（修正後）
self.robot = self.scene.add_entity(...)
self.scene.build()  # ← 最初に実行
self._setup_robot_info()
self._setup_pd_control()
self._set_initial_pose()
```

### 2. **_set_initial_pose() が何もしていなかった**

```python
# ❌ 以前
def _set_initial_pose(self):
    initial_angles = np.array([...])  # 定義するだけ
    print("✓ Initial pose set (full body)")  # 実際には何もしない！
```

```python
# ✅ 修正後
def _set_initial_pose(self):
    initial_angles = np.array([...])
    
    print("Setting initial pose with hard reset...")
    # 硬いリセット: 複数回繰り返して確実に設定
    for step in range(100):
        self.robot.set_dofs_position(initial_angles, self.dofs_idx_local)
        self.scene.step()  # ← シミュレーションを進める！
    
    # 落ち着かせるためにもう少し待つ
    print("Settling initial pose...")
    for _ in range(200):
        self.robot.control_dofs_position(initial_angles, self.dofs_idx_local)
        self.scene.step()
```

### 3. **SteppingController が初期化されていないのに呼び出されていた**

```python
# ❌ 以前
def step_simulation(self):
    self.timer.time = self.time  # ← self.timer が None！
    self.stepping_controller.update(...)  # ← stepping_controller が None！
```

```python
# ✅ 修正後
def step_simulation(self):
    if hasattr(self, 'stepping_controller') and self.stepping_controller:
        # SteppingController が初期化されている場合のみ実行
        self.timer.time = self.time
        self.stepping_controller.update(...)
    else:
        # 初期化されるまでは、単に関節角度を保持するだけ
        pass
```

## ✅ 修正内容

### 変更1: scene.build() の順序
```python
# hrp2_walking_sim.py の _init_genesis() メソッド

self.robot = self.scene.add_entity(
    gs.morphs.URDF(file=robot_path,
                   pos=(0.0, 0.0, 0.9),  # 高めにスポーン
                   fixed=False),
)

# ★ ここでscene.build()を実行
self.scene.build()

# その後に設定
self._setup_robot_info()
self._setup_pd_control()
self._set_initial_pose()
```

### 変更2: _set_initial_pose() の実装
```python
def _set_initial_pose(self):
    """初期姿勢を設定"""
    initial_angles = np.array([...])
    
    # ハードリセット (100ステップ)
    for step in range(100):
        self.robot.set_dofs_position(initial_angles, self.dofs_idx_local)
        self.scene.step()  # ← 重要: シミュレーションを進める
    
    # 落ち着かせる (200ステップ)
    for _ in range(200):
        self.robot.control_dofs_position(initial_angles, self.dofs_idx_local)
        self.scene.step()
```

### 変更3: step_simulation() の安全性向上
```python
def step_simulation(self):
    # SteppingController が初期化されている場合のみ実行
    if hasattr(self, 'stepping_controller') and self.stepping_controller:
        self.stepping_controller.update(...)
    
    self.scene.step()
    self.time += self.dt
```

### 変更4: __init__() で SteppingController を初期化
```python
def __init__(self, dt=0.02, render=True):
    # ...
    self.stepping_controller = None  # 後で初期化される
    self.timer = None
    self.centroid = None
    self.base = None
    self.feet = None
    # ...
```

## 🎯 効果

| 問題 | 原因 | 修正後 |
|------|------|--------|
| ロボットが倒れる | scene.build() の前に設定が無視された | ✓ 設定が反映される |
| PD制御が機能しない | _set_initial_pose() が何もしていない | ✓ 100ステップ + 200ステップで安定化 |
| SteppingController エラー | 初期化前に呼び出し | ✓ 条件チェック追加 |

## 🧪 テスト方法

```bash
# 初期姿勢を保持したままシミュレーション
python hrp2_walking_sim.py --duration 5 --no-viewer
```

**確認項目**:
- ✓ ロボットが立ったまま倒れない
- ✓ 腕がブラブラ揺れない
- ✓ 0～3秒で初期姿勢に落ち着く
- ✓ 以降、姿勢を保持

## 📊 初期化の流れ（修正後）

```
1. self.robot = URDF(pos=(0,0,0.9)) を追加
   ↓
2. self.scene.build() ← ★ 重要
   ↓
3. _setup_robot_info() ... 関節情報取得
   ↓
4. _setup_pd_control() ... PD制御パラメータ設定
   ↓
5. _set_initial_pose()
   - 100ステップ: hard reset (set_dofs_position)
   - 200ステップ: settling (control_dofs_position + scene.step)
   ↓
6. 安定した状態で歩行を開始可能
```

## 💡 次のステップ

1. **歩行テスト** (generate_walking_plan() 実行)
   ```bash
   python hrp2_walking_sim.py --stride 0.2 --duration 10
   ```

2. **パラメータ微調整**
   - 初期高さ: pos=(0, 0, 0.9) を調整
   - PDゲイン: kp, kv 値の最適化
   - ステップ数: 100/200 を調整

3. **歩行軌道の検証**
   - ログ出力の追加
   - 足の位置・姿勢の記録
   - グラフ化

---

**修正完了日**: 2026-05-18
**ステータス**: ✅ ロボット姿勢安定化完了
