"""
HRP2 Walking Simulation with Footstep Planning
Genesis上でロボットが決定論的な歩行計画に従って歩く
"""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "irsl_rl"))

import genesis as gs
import torch

from footstep_planner import FootstepPlanner, Step, Footstep, Param, Ground
from stepping_controller import SteppingController, Timer, Centroid, Base, Foot
from hrp2_footplane import generate_5step_walking_plan


class HRP2WalkingSimulator:
    """HRP2 歩行シミュレータ"""
    
    def __init__(self, dt=0.02, render=True):
        """
        初期化
        
        Args:
            dt: シミュレーション時間ステップ [s]
            render: ビューアーを表示するか
        """
        self.dt = dt
        self.render = render
        self.scene = None
        self.robot = None
        self.viewer = None
        self.time = 0.0
        
        # ロボット固有のパラメータ
        self.joint_names = []
        self.dofs_idx_local = None
        self.num_motors = 0
        self.joint_indices = {}
        
        # 歩行計画
        self.footstep = None
        self.param = None
        self.current_step = 0
        self.step_progress = 0.0  # 0 ~ 1
        
        # SteppingController
        self.stepping_controller = None
        self.timer = None
        self.centroid = None
        self.base = None
        self.feet = None
        
        # 目標位置・姿勢
        self.target_joint_angles = None
        
        self._init_genesis()
    
    def _init_genesis(self):
        """Genesis を初期化"""
        print("Initializing Genesis...")
        gs.init(backend=gs.cuda)
        
        # Scene 作成オプション
        scene_kwargs = {
            "sim_options": gs.options.SimOptions(
                dt=self.dt,
                gravity=[0, 0, -9.81],
            ),
        }
        
        if self.render:
            scene_kwargs["show_viewer"] = True
        
        self.scene = gs.Scene(**scene_kwargs)
        
        # 地面の追加
        self.ground = self.scene.add_entity(gs.morphs.Plane())
        
        # HRP2 ロボットを追加
        robot_path = os.path.join(os.path.dirname(__file__), "hrp2_description/HRP2_genesis.urdf")
        print(f"Loading URDF from: {robot_path}")
        
        try:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(file=robot_path,
                               pos=(0.0, 0.0, 0.71),  # ハーフシッティングに合わせた初期高さ
                               fixed=False
                               ),
            )
            print(f"✓ Successfully loaded HRP2 robot")
        except Exception as e:
            print(f"✗ Error loading URDF: {e}")
            raise
        
        # scene.build() を実行
        self.scene.build()
        
        # ロボット情報を取得
        self._setup_robot_info()
        
        # PD制御パラメータを設定
        self._setup_pd_control()
        
        # 初期姿勢を設定
        self._set_initial_pose()
        
        print(f"✓ Genesis initialized with {self.num_motors} DOFs")
        
        if self.render and hasattr(self.scene, 'viewer'):
            self.viewer = self.scene.viewer
    
    def _setup_robot_info(self):
        """ロボット情報をセットアップ（hrp2_train.py の並び順と完全同期）"""
        self.joint_names = [
            # 右脚 (6 DOF)
            "RLEG_JOINT0", "RLEG_JOINT1", "RLEG_JOINT2", "RLEG_JOINT3", "RLEG_JOINT4", "RLEG_JOINT5",
            # 左脚 (6 DOF)
            "LLEG_JOINT0", "LLEG_JOINT1", "LLEG_JOINT2", "LLEG_JOINT3", "LLEG_JOINT4", "LLEG_JOINT5",
            # 体幹 (2 DOF)
            "CHEST_JOINT0", "CHEST_JOINT1",
            # 頭部 (2 DOF)
            "HEAD_JOINT0", "HEAD_JOINT1",
            # 右腕 (7 DOF)
            "RARM_JOINT0", "RARM_JOINT1", "RARM_JOINT2", "RARM_JOINT3", "RARM_JOINT4", "RARM_JOINT5", "RARM_JOINT6",
            # 左腕 (7 DOF)
            "LARM_JOINT0", "LARM_JOINT1", "LARM_JOINT2", "LARM_JOINT3", "LARM_JOINT4", "LARM_JOINT5", "LARM_JOINT6",
        ]
        
        self.dofs_idx_local = []
        self.joint_indices = {}
        
        for joint_name in self.joint_names:
            try:
                dof_idx = self.robot.get_joint(joint_name).dof_idx_local
                self.dofs_idx_local.append(dof_idx)
                self.joint_indices[joint_name] = dof_idx
            except Exception as e:
                print(f"  Warning: Could not get joint {joint_name}: {e}")
        
        self.dofs_idx_local = np.array(self.dofs_idx_local)
        self.num_motors = len(self.dofs_idx_local)
        print(f"✓ Found {self.num_motors} motor DOFs")
    
    def _setup_pd_control(self):
        """PD制御パラメータを設定（hrp2_train.py の強固なゲインに設定）"""
        try:
            if not self.robot or not hasattr(self, 'dofs_idx_local'):
                return
            
            # 全身一律で高ゲインを適用
            kp_values = np.ones(self.num_motors, dtype=np.float32) * 2000.0
            kv_values = np.ones(self.num_motors, dtype=np.float32) * 100.0
            
            self.robot.set_dofs_kp(kp_values, self.dofs_idx_local)
            self.robot.set_dofs_kv(kv_values, self.dofs_idx_local)
            
            # トルクリミットの設定
            force_upper = np.ones(self.num_motors, dtype=np.float32) * 300.0
            force_lower = -force_upper
            self.robot.set_dofs_force_range(force_lower, force_upper, self.dofs_idx_local)
            
            print(f"✓ PD control configured (kp=2000.0, kd=100.0)")
        except Exception as e:
            print(f"Warning: Could not setup PD control: {e}")
    
    def _set_initial_pose(self):
        """初期姿勢を設定（hrp2_train.py のハーフシッティング姿勢）"""
        try:
            if not self.robot or not hasattr(self, 'dofs_idx_local'):
                return
            
            # 股ピッチ=-0.4, 膝=0.8, 足首ピッチ=-0.4
            initial_angles = np.array([
                # 右脚: 姿勢
                0.0, 0.0, -0.4, 0.8, -0.4, 0.0,
                # 左脚: 姿勢
                0.0, 0.0, -0.4, 0.8, -0.4, 0.0,
                # 腰・頭・両腕はニュートラル
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32)
            
            print("Setting initial pose (Half-Sitting)...")
            for step in range(100):
                self.robot.set_dofs_position(initial_angles, self.dofs_idx_local)
                self.scene.step()
            
            for settle_step in range(200):
                self.robot.control_dofs_position(initial_angles, self.dofs_idx_local)
                self.scene.step()
            
            print("✓ Initial pose set and settled")
        except Exception as e:
            print(f"Warning: Could not set initial pose: {e}")
    
    def _setup_stepping_controller(self):
        """SteppingController を初期化"""
        self.stepping_controller = SteppingController()
        self.timer = Timer()
        self.centroid = Centroid()
        self.base = Base()
        self.feet = [Foot(), Foot()]  # [左足, 右足]
        
        # 初期位置を設定（ワールド座標系における足底の初期位置）
        self.feet[0].pos_ref = np.array([0.0, 0.1, 0.0])   # 左足 (spacing 0.2)
        self.feet[1].pos_ref = np.array([0.0, -0.1, 0.0])  # 右足
        
        # 重要: コントローラ内部の仕様に合わせて2ステップ先まで空のインスタンスを予約
        self.footstep_buffer = Footstep(steps=[Step(), Step()])
        
        print("✓ SteppingController initialized")
    
    def generate_walking_plan(self, stride=0.2, sway=0.0, turn=0.0, 
                              spacing=0.2, com_height=0.71, T=1.0, 
                              duration_per_step=0.8):
        """歩行計画を生成"""
        print("\nGenerating walking plan...")
        self.footstep, self.param = generate_5step_walking_plan(
            stride=stride,
            sway=sway,
            turn=turn,
            spacing=spacing,
            com_height=com_height,
            T=T,
            duration_per_step=duration_per_step
        )
        
        # 参考 DCM と ZMP 軌道を事前にバックエンドで生成
        planner = FootstepPlanner()
        planner.generate_dcm(self.param, self.footstep)
        
        print(f"✓ Generated {len(self.footstep.steps)} steps")
        
        # コントローラの初期化
        self._setup_stepping_controller()
        
        # 最初のステップの状態を反映
        if len(self.footstep.steps) > 0:
            st0 = self.footstep.steps[0]
            self.centroid.zmp_ref = st0.zmp.copy()
            self.centroid.dcm_ref = st0.dcm.copy()
            self.centroid.zmp_target = st0.zmp.copy()
            self.centroid.dcm_target = st0.dcm.copy()
        
        self.current_step = 0
        self.step_progress = 0.0
    
    def step_simulation(self):
        """シミュレーションを1ステップ進める"""
        try:
            if hasattr(self, 'stepping_controller') and self.stepping_controller:
                self.timer.time = self.time
                
                # 決定論的（オープンループ）に歩行させるため、前回の出力を今回の参照としてフィードバック
                self.centroid.dcm_ref = self.centroid.dcm_target.copy()
                self.centroid.zmp_ref = self.centroid.zmp_target.copy()
                
                # 軌道更新
                self.stepping_controller.update(
                    self.timer, 
                    self.param, 
                    self.footstep,
                    self.footstep_buffer,
                    self.centroid,
                    self.base,
                    self.feet
                )
                
                # 逆運動学の計算と指令
                self._update_joint_targets_from_feet()
            
        except Exception as e:
            print(f"Warning in stepping controller: {e}")
        
        self.scene.step()
        self.time += self.dt
    
    def _update_joint_targets_from_feet(self):
        """足の目標位置・姿勢から逆運動学(IK)を計算して関節角を一括制御"""
        try:
            if not self.robot or not hasattr(self, 'dofs_idx_local'):
                return
            
            # 各足のワールド座標目標
            left_pos = self.feet[0].pos_ref
            right_pos = self.feet[1].pos_ref
            
            # DCMベースの目標重心位置から腰の推定目標位置を算出
            base_pos_ref = self.centroid.dcm_target - np.array([0.0, 0.0, self.param.com_height])
            
            # 腰から見た各足の相対位置
            left_rel_pos = left_pos - base_pos_ref
            right_rel_pos = right_pos - base_pos_ref
            
            # ----------------- 左脚の幾何IK -----------------
            left_H = -left_rel_pos[2]
            # 初期高さ 0.71m のときに膝がちょうど 0.8rad になるリンク幾何モデル
            left_knee = 2.0 * np.arccos(np.clip(left_H / 0.77, -1.0, 1.0))
            
            # 姿勢維持（平行リンク近似: 股ピッチと足首ピッチは膝の半分ずつ受け持つ）
            left_hip_pitch = -0.5 * left_knee
            left_ankle_pitch = -0.5 * left_knee
            
            # X方向（前後移動）によるピッチ補正
            left_hip_pitch += -left_rel_pos[0] / 0.71
            left_ankle_pitch += -left_rel_pos[0] / 0.71
            
            # Y方向（左右揺れ）によるロール補正 (デフォルトの足幅 0.1m からの変位)
            left_hip_roll = (left_rel_pos[1] - 0.1) / 0.71
            left_ankle_roll = -(left_rel_pos[1] - 0.1) / 0.71
            left_hip_yaw = self.feet[0].angle_ref[2]
            
            # ----------------- 右脚の幾何IK -----------------
            right_H = -right_rel_pos[2]
            right_knee = 2.0 * np.arccos(np.clip(right_H / 0.77, -1.0, 1.0))
            
            right_hip_pitch = -0.5 * right_knee
            right_ankle_pitch = -0.5 * right_knee
            
            right_hip_pitch += -right_rel_pos[0] / 0.71
            right_ankle_pitch += -right_rel_pos[0] / 0.71
            
            right_hip_roll = (right_rel_pos[1] - (-0.1)) / 0.71
            right_ankle_roll = -(right_rel_pos[1] - (-0.1)) / 0.71
            right_hip_yaw = self.feet[1].angle_ref[2]
            
            # -----------------------------------------------
            # 関節角度配列の構築 (hrp2_train.py の joint_names の順序と完全同期)
            target_angles = np.array([
                # 右脚 (6 DOF)
                right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll,
                # 左脚 (6 DOF)
                left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll,
                # 腰 (2 DOF)
                0.0, 0.0,
                # 頭 (2 DOF)
                0.0, 0.0,
                # 右腕 (7 DOF)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # 左腕 (7 DOF)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32)
            
            self.set_all_joint_targets(target_angles)
            
        except Exception as e:
            print(f"IK Error: {e}")
    
    def set_all_joint_targets(self, angles):
        """すべての制御対象関節の目標角度を一括設定"""
        try:
            if self.robot and len(angles) == self.num_motors:
                angles = np.asarray(angles, dtype=np.float32)
                self.robot.control_dofs_position(angles, self.dofs_idx_local)
        except Exception as e:
            print(f"Warning: Could not set joint targets: {e}")
            
    def render_step(self):
        """画面を描画"""
        try:
            if self.render and self.viewer:
                self.viewer.render()
        except:
            pass
    
    def run(self, duration=10.0):
        """シミュレーションを実行"""
        print(f"\nRunning simulation for {duration} seconds...")
        
        num_steps = int(duration / self.dt)
        for i in range(num_steps):
            self.step_simulation()
            self.render_step()
            
            if i % 100 == 0:
                if self.feet:
                    print(f"Time: {self.time:.2f}s | Left Foot Z: {self.feet[0].pos_ref[2]:.3f} | Right Foot Z: {self.feet[1].pos_ref[2]:.3f}")
            
            if self.time > duration:
                break
        
        print(f"\n✓ Simulation completed! Total time: {self.time:.2f} s")


def main():
    parser = argparse.ArgumentParser(description='HRP2 Walking Simulation')
    parser.add_argument('--stride', type=float, default=0.15, help='Stride distance (m)')
    parser.add_argument('--sway', type=float, default=0.02, help='Sway (m)')
    parser.add_argument('--turn', type=float, default=0.0, help='Turn angle (rad)')
    parser.add_argument('--spacing', type=float, default=0.2, help='Foot spacing (m)')
    parser.add_argument('--com-height', type=float, default=0.71, help='CoM height (m)')
    parser.add_argument('--T', type=float, default=0.4, help='Time constant (s)')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration (s)')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step (s)')
    parser.add_argument('--no-viewer', action='store_true', help='Disable viewer')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HRP2 Walking Simulation with Footstep Planning")
    print("="*70)
    
    sim = HRP2WalkingSimulator(dt=args.dt, render=not args.no_viewer)
    
    sim.generate_walking_plan(
        stride=args.stride,
        sway=args.sway,
        turn=args.turn,
        spacing=args.spacing,
        com_height=args.com_height,
        T=args.T,
        duration_per_step=0.8
    )
    
    sim.run(duration=args.duration)
    print("="*70)


if __name__ == "__main__":
    main()