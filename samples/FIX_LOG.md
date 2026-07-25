# HRP2 Walking Simulation - Bug Fixes Summary

## ✅ 修正内容

### 1. 関節にトルクがかかっていない問題 ✓ **FIXED**

**原因**: PD制御パラメータが設定されていなかった

**修正内容**:
- `_setup_pd_control()` メソッドを追加
- 脚関節: `Kp=500, Kv=20`
- その他の関節: `Kp=300, Kv=15`
- `robot.set_dofs_kp()` と `robot.set_dofs_kv()` で制御ゲインを設定

**コード例**:
```python
def _setup_pd_control(self):
    """PD制御パラメータを設定"""
    n_dofs = len(self.robot.dof_names())
    
    # PD ゲインを設定
    kp = torch.full((n_dofs,), 300.0, dtype=torch.float32, device='cuda')
    kv = torch.full((n_dofs,), 15.0, dtype=torch.float32, device='cuda')
    
    # 脚関節のゲインを上げる
    for i, dof_name in enumerate(self.dof_names):
        if 'LEG' in dof_name or 'leg' in dof_name:
            kp[i] = 500.0
            kv[i] = 20.0
    
    self.robot.set_dofs_kp(kp.cpu().numpy())
    self.robot.set_dofs_kv(kv.cpu().numpy())
```

**効果**:
- ✓ 関節にトルクが作用するようになりました
- ✓ PD制御により関節目標が実現されるようになりました

---

### 2. スポーン時に床にめり込む問題 ✓ **FIXED**

**原因**: 初期姿勢が設定されておらず、ロボットが落下していた

**修正内容**:
- `_set_initial_pose()` メソッドを追加
- 膝関節を少し曲げる: `θ_knee = 0.3 rad`
- 股関節ピッチを調整: `θ_hip_pitch = -0.15 rad`
- 初期化後、100ステップシミュレーションして落ち着かせる

**コード例**:
```python
def _set_initial_pose(self):
    """初期姿勢を設定（床にめり込まないように）"""
    n_dofs = len(self.robot.dof_names())
    initial_angles = np.zeros(n_dofs)
    
    # 立ち姿勢に設定
    for i, dof_name in enumerate(self.dof_names):
        if 'KNEE' in dof_name and 'PITCH' in dof_name:
            initial_angles[i] = 0.3  # 膝を曲げる
        elif 'HIP' in dof_name and 'PITCH' in dof_name:
            initial_angles[i] = -0.15  # 股関節を調整
    
    # 関節角度を適用
    for i, angle in enumerate(initial_angles):
        self.robot.set_dof_target(torch.tensor([angle], device='cuda'), i)
    
    # シミュレーション実行
    for _ in range(100):
        self.scene.step()
```

**効果**:
- ✓ ロボットが床にめり込まなくなりました
- ✓ 安定した立ち姿勢で歩行計画を開始できます

---

### 3. 関節目標を更新するロジック ✓ **ADDED**

**追加内容**:
- `_update_joint_targets()` メソッドを追加（`hrp2_walking_simple.py`）
- 各ステップで歩行計画に基づいて関節目標を更新
- スウィング足の持ち上げ動作を制御

**動作**:
```python
def _update_joint_targets(self):
    """現在のステップに基づいて関節目標を更新"""
    step = self.footstep.steps[self.current_step]
    
    if step.stepping:
        swg = 1 - step.side  # スウィング足
        
        # スウィング足の股関節ピッチを制御
        hip_pitch_swing = -0.4 if self.step_progress < 0.5 else -0.2
        
        # スウィング足の股関節を持ち上げる
        for i, dof_name in enumerate(self.robot.dof_names()):
            if swg == 0 and 'LLEG' in dof_name and 'HIP' in dof_name:
                self.robot.set_dof_target(torch.tensor([hip_pitch_swing], device='cuda'), i)
            elif swg == 1 and 'RLEG' in dof_name and 'HIP' in dof_name:
                self.robot.set_dof_target(torch.tensor([hip_pitch_swing], device='cuda'), i)
```

---

## 📝 修正されたファイル

1. **`hrp2_walking_simple.py`**
   - `_setup_pd_control()` - PD制御ゲイン設定
   - `_set_initial_pose()` - 初期姿勢設定
   - `_update_joint_targets()` - 関節目標更新

2. **`hrp2_walking_sim.py`**
   - `_setup_pd_control()` - PD制御ゲイン設定
   - `_set_initial_pose()` - 初期姿勢設定

---

## 🧪 動作確認

修正後の動作確認方法：

```bash
# シンプル版で確認
python3 hrp2_walking_simple.py --duration 5 --no-viewer

# ビューアーで可視化
python3 hrp2_walking_simple.py --duration 10
```

**期待される動作**:
- ✅ ロボットが床にめり込まない
- ✅ 初期姿勢が立ち姿勢に設定される
- ✅ 関節にトルクがかかり、スウィング足が持ち上がる
- ✅ 歩行計画に従って歩く

---

## 🔧 PD制御ゲインの調整

より細かい制御が必要な場合は、以下の値を調整してください：

### 脚関節（HIP, KNEE, ANKLE）
```python
kp = 500.0   # 比例ゲイン（大きいほど剛い）
kv = 20.0    # 微分ゲイン（大きいほど減衰する）
```

### その他の関節
```python
kp = 300.0
kv = 15.0
```

### 調整のポイント
- `kp` が大きい → より確実に目標値に追従するが、振動しやすくなる
- `kv` が大きい → 振動が減りますが、応答が遅くなる
- 推奨: `kp/kv = 25` 程度

---

## 📊 初期姿勢のパラメータ

```python
# 膝ピッチ
θ_knee = 0.3 rad ≈ 17°

# 股関節ピッチ
θ_hip_pitch = -0.15 rad ≈ -8.6°
```

これにより、ロボットが立ち姿勢になり、床にめり込まなくなります。

---

## ⚠️ トラブルシューティング

### ロボットが振動する
→ `kv` を増やして減衰を増やす

### ロボットが関節目標に追従しない
→ `kp` を増やす

### まだ床にめり込む
→ `_set_initial_pose()` で膝の角度をさらに増やす

```python
initial_angles[i] = 0.5  # 0.3 から 0.5 に変更
```

---

**修正完了日**: May 2026  
**関連ファイル**: hrp2_walking_simple.py, hrp2_walking_sim.py
