## HRP2 Walking Simulator - SteppingController 統合版

### 修正内容

`hrp2_walking_sim.py` を `hrp2_eval_gs.py` を参考にして、`FootstepPlanner` と `SteppingController` を使用した実際のロボット歩行軌道生成に更新しました。

### 主な変更点

#### 1. **Imports に SteppingController を追加**
```python
from stepping_controller import SteppingController, Timer, Centroid, Base, Foot
```

#### 2. **新しいメソッド: `_setup_stepping_controller()`**
```python
def _setup_stepping_controller(self):
    """SteppingController を初期化"""
    self.stepping_controller = SteppingController()
    self.timer = Timer()
    self.centroid = Centroid()
    self.base = Base()
    self.feet = [Foot(), Foot()]  # [左足, 右足]
    self.footstep_buffer = Footstep()
```

- SteppingController のインスタンス化
- タイマー、重心、ベース、足情報の初期化
- 足のバッファ化されたステップ情報を管理

#### 3. **改善された `step_simulation()` メソッド**

以前:
```python
# 簡略版：階段関数的に関節を動かす
if step.stepping:
    swg = 1 - step.side
    if swg == 0:
        if self.step_progress < 0.5:
            self.set_joint_target("left_hip_pitch", -0.3 * self.step_progress * 4)
        else:
            self.set_joint_target("left_hip_pitch", -0.3 * (1 - self.step_progress) * 4)
```

現在:
```python
# SteppingController で足の軌道を生成
self.stepping_controller.update(
    self.timer, 
    self.param, 
    self.footstep,
    self.footstep_buffer,
    self.centroid,
    self.base,
    self.feet
)

# 足の目標位置から関節角度を計算・設定
self._update_joint_targets_from_feet()
```

#### 4. **新しいメソッド: `_update_joint_targets_from_feet()`**

足の目標位置・姿勢から関節角度を計算して設定：

```python
def _update_joint_targets_from_feet(self):
    """足の目標位置・姿勢から関節角度を計算して設定"""
    # SteppingController が生成した足の目標位置を取得
    left_pos = self.feet[0].pos_ref
    right_pos = self.feet[1].pos_ref
    
    # 高さに基づいて膝と足首の角度を計算
    left_knee_angle = self._height_to_knee_angle(left_pos[2])
    right_knee_angle = self._height_to_knee_angle(right_pos[2])
    
    # X方向の位置に基づいて Hip Pitch を調整
    self.set_joint_target("left_hip_pitch", -0.3 * left_pos[0])
    self.set_joint_target("right_hip_pitch", -0.3 * right_pos[0])
```

#### 5. **新しいメソッド: `_height_to_knee_angle()`**

簡略版の逆運動学で高さを関節角度に変換：

```python
def _height_to_knee_angle(self, height, leg_length=0.8):
    """高さから膝の角度を計算（簡略版）"""
    standing_height = 0.8  # 立ち状態での脚の長さ
    
    if height >= standing_height:
        return 0.0  # 膝をまっすぐ
    else:
        # 屈曲を計算（簡略版）
        deficit = standing_height - height
        knee_angle = 0.5 * deficit
        return knee_angle
```

### 動作フロー

```
1. generate_walking_plan()
   ├── FootstepPlanner で5ステップの歩行計画を生成
   └── SteppingController を初期化

2. step_simulation() （毎ステップ実行）
   ├── タイマーを更新（self.timer.time = self.time）
   ├── SteppingController.update() で足の軌道を生成
   │   ├── DCM/ZMP ベースのステップ調整
   │   ├── スウィング足のサイクロイド軌道生成
   │   └── 足の目標位置・姿勢を更新（self.feet）
   ├── _update_joint_targets_from_feet() で関節角度を計算
   │   ├── 高さから膝の角度を計算
   │   ├── X位置から Hip Pitch を計算
   │   └── 足首角度を設定
   └── self.scene.step() でシミュレーション実行
```

### 利用例

```bash
# 基本的な実行
python3 hrp2_walking_sim.py --duration 10 --stride 0.2

# パラメータ指定
python3 hrp2_walking_sim.py \
    --stride 0.25 \
    --spacing 0.18 \
    --com-height 0.75 \
    --duration 15 \
    --no-viewer

# ビューアー表示
python3 hrp2_walking_sim.py --duration 10
```

### 関連クラス・メソッド

#### FootstepPlanner
- `plan()` - 足の配置計画
- `align_to_ground()` - 地面に足を合わせる
- `generate_dcm()` - DCM/ZMP 生成

#### SteppingController
- `update()` - リアルタイムの足の軌道更新
  - DCM/ZMP ベースのステップ調整
  - スウィング足のサイクロイド軌道生成
  - ベース向きの計算

#### データ構造
- `Timer` - 現在時刻
- `Centroid` - 重心（DCM/ZMP）情報
- `Base` - ロボットベース（ボディ）情報
- `Foot` - 足の情報（位置・姿勢・接触フラグ）
- `Footstep` - 複数ステップの歩行計画

### トルク制御の確認

PD制御パラメータが `_setup_pd_control()` で設定されています：

```python
kp = 500.0  # 脚関節の比例ゲイン
kv = 20.0   # 脚関節の微分ゲイン
```

- Genesis は内部的に PD 制御で関節をドライブ
- `set_dof_target()` で目標角度を設定
- Genesis が自動的にトルク計算

### 次のステップ

1. **より精密な逆運動学**
   - 現在は簡略版（高さのみ考慮）
   - 完全な6DOF IK を実装可能

2. **足首の接触判定**
   - `Foot.contact_ref` を使用した接触判定
   - 地面反力のシミュレーション

3. **ダイナミクス統合**
   - ロボットの重力補償
   - 床反力を利用した安定化制御

4. **パラメータ最適化**
   - 脚長に基づいた逆運動学の調整
   - PD ゲインの最適化
