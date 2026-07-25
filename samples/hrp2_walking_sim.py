"""
HRP2 Walking Simulation with Footstep Planning
Genesis上でロボットが決定論的な歩行計画に従って歩く
"""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import matplotlib
matplotlib.use('Agg')  # ← 最上部に追加（GUI不要なバックエンド）
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "irsl_rl"))

import genesis as gs
import torch

# ★修正: footstep_planner からのインポート（generate_5step_walking_plan は除外）
from footstep_planner import FootstepPlanner, Step, Footstep, Param, Ground
# ★修正: hrp2_footplane から関数をインポート
from hrp2_footplane import generate_5step_walking_plan
from stepping_controller import SteppingController, Timer, Centroid, Base, Foot


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
        
        # 内部状態（重心の実際の位置）の追加
        self.com_pos = None
        
        # 目標位置・姿勢
        self.target_joint_angles = None
        
        # 記録用の変数
        self.time_history = []
        self.com_history = []
        self.zmp_history = []
        self.lfoot_history = []
        self.rfoot_history = []
        self.dcm_history = []
        
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
                               pos=(0.0, 0.0, 0.70),  # ハーフシッティングに合わせた初期高さ
                               fixed=False
                               ),
            )
            print(f"✓ Successfully loaded HRP2 robot")
        except Exception as e:
            print(f"✗ Error loading URDF: {e}")
            raise
        
        # scene.build() を実行
        self.scene.build()

        input("Press Enter to start setup...")

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
        
        # 【追加】IKターゲットとなるリンクオブジェクトの取得
        # ※お使いのURDFのリンク定義名（LLEG_LINK5 や CHEST_LINK0等）に合わせて適宜修正してください
        try:
            self.left_foot_link = self.robot.get_link("LLEG_LINK5")
            self.right_foot_link = self.robot.get_link("RLEG_LINK5")
            self.base_link = self.robot.get_link("CHEST_LINK0")  # 腰・骨盤の基準リンク
            print("✓ Successfully retrieved links for IK")
        except Exception as e:
            print(f"✗ Warning: Could not get links for IK from URDF: {e}")
            print("  Please verify link names (e.g., LLEG_LINK5) in your HRP2 URDF file.")
    
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
    
    def generate_walking_plan(self, stride=0.05, sway=0.0, turn=0.0, 
                              spacing=0.2, com_height=0.65, T=1.0, 
                              duration_per_step=0.8):
        """歩行計画を生成"""
        print("\n" + "="*70)
        print("Generating walking plan...")
        print(f"Parameters:")
        print(f"  stride: {stride} m")
        print(f"  spacing: {spacing} m")
        print(f"  CoM height: {com_height} m")
        print(f"  Time constant T: {T} s")
        print("="*70)
        
        # ★修正: 関数を直接呼び出す（クラスメソッドではなく）
        self.footstep, self.param = generate_5step_walking_plan(
            stride=stride,
            sway=sway,
            turn=turn,
            spacing=spacing,
            com_height=com_height,
            T=T,
            duration_per_step=duration_per_step
        )
        
        # 【デバッグ】生成されたステップを確認
        print(f"\n✓ Generated {len(self.footstep.steps)} steps")
        print("\nFootstep Details:")
        print("-" * 70)
        for i, step in enumerate(self.footstep.steps):
            print(f"Step {i:2d}: ZMP=[{step.zmp[0]:.4f}, {step.zmp[1]:.4f}, {step.zmp[2]:.4f}]  " +
                  f"DCM=[{step.dcm[0]:.4f}, {step.dcm[1]:.4f}, {step.dcm[2]:.4f}]  " +
                  f"Stepping={getattr(step, 'stepping', 'N/A')}")
        print("-" * 70)
        
        # ステップの設定
        for i, step in enumerate(self.footstep.steps):
            if i == 0:
                step.stepping = False
            else:
                step.stepping = True
        
        # DCMとZMP軌道を生成
        planner = FootstepPlanner()
        planner.generate_dcm(self.param, self.footstep)
        
        print(f"✓ Parameters:")
        print(f"  CoM Height: {self.param.com_height:.4f} m")
        print(f"  Time Constant T: {self.param.T:.4f} s")
        
        # ★【新規】フットステップ計画のデバッグプロットを生成
        print("\nGenerating footstep debug plot...")
        try:
            self.plot_footstep_generation_debug("footstep_debug.png")
        except Exception as e:
            print(f"Warning: Could not generate footstep debug plot: {e}")
        
        # コントローラの初期化
        self._setup_stepping_controller()
        
        # 最初のステップの状態を反映
        if len(self.footstep.steps) > 0:
            st0 = self.footstep.steps[0]
            self.centroid.zmp_ref = st0.zmp.copy()
            self.centroid.dcm_ref = st0.dcm.copy()
            self.centroid.zmp_target = st0.zmp.copy()
            self.centroid.dcm_target = st0.dcm.copy()
            
            self.com_pos = np.array([st0.dcm[0], st0.dcm[1], com_height])
        
        self.current_step = 0
        self.step_progress = 0.0
        print("="*70 + "\n")
    
    def step_simulation(self):
        """シミュレーションを1ステップ進める"""
        try:
            if hasattr(self, 'stepping_controller') and self.stepping_controller:
                # 1. タイマーの更新（シミュレーションの絶対時間）
                self.timer.time = self.time
                
                # 2. 目標DCM（dcm_target）から実際の重心位置（CoM）をオイラー積分で進展させる
                com_dot = (self.centroid.dcm_target - self.com_pos) / self.param.T
                self.com_pos += com_dot * self.dt
                
                # 積分結果から現在の実際のDCM参照値をフィードバック
                self.centroid.dcm_ref = self.com_pos.copy()
                self.centroid.zmp_ref = self.centroid.zmp_target.copy()
                
                # 3. 軌道更新 (歩行計画が残っている間だけ実行)
                if len(self.footstep.steps) > 0:
                    self.stepping_controller.update(
                        self.timer, 
                        self.param, 
                        self.footstep,
                        self.footstep_buffer,
                        self.centroid,
                        self.base,
                        self.feet
                    )
                
                # 4. 逆運動学(Genesis IK)の計算と指令
                self._update_joint_targets_from_feet()
                
                # 【新規】データ記録
                self._record_state()
            
        except Exception as e:
            print(f"Warning in step_simulation: {e}")
        
        self.scene.step()
        self.time += self.dt
    
    def _record_state(self):
        """シミュレーション状態を記録"""
        self.time_history.append(self.time)
        self.com_history.append(self.com_pos.copy())
        self.zmp_history.append(self.centroid.zmp_target.copy())
        self.lfoot_history.append(self.feet[0].pos_ref.copy())
        self.rfoot_history.append(self.feet[1].pos_ref.copy())
        self.dcm_history.append(self.centroid.dcm_target.copy())
    
    def _update_joint_targets_from_feet(self):
        """IKと直接制御（path planningなし）"""
        if self.robot is None:
            return

        try:
            right_foot_pos = np.array(self.feet[0].pos_ref, dtype=np.float32)
            left_foot_pos = np.array(self.feet[1].pos_ref, dtype=np.float32)
            foot_quat = np.array([0, 0, 0, 1], dtype=np.float32)  # [x, y, z, w]
            
            # IK計算のみ（時間がかかる処理）
            qpos_right = self.robot.inverse_kinematics(
                link=self.right_foot_link,
                pos=right_foot_pos,
                quat=foot_quat
            )
            
            qpos_left = self.robot.inverse_kinematics(
                link=self.left_foot_link,
                pos=left_foot_pos,
                quat=foot_quat
            )
            
            # 合成
            target_qpos = self._synthesize_ik_results(qpos_right, qpos_left)
            
            # ★ scene.step() は呼ばずに、制御指令のみ
            self.robot.control_dofs_position(target_qpos, self.dofs_idx_local)
            
        except Exception as e:
            print(f"[IK Error]: {e}")
    
    def _synthesize_ik_results(self, qpos_right, qpos_left):
        """右足・左足のIK結果を合成して、全関節の目標角度配列を生成
        
        Args:
            qpos_right: 右足IKの結果（30 DOF 全体）
            qpos_left: 左足IKの結果（30 DOF 全体）
        
        Returns:
            target_qpos: 全関節の目標角度（30 DOF）
        """
        # 初期値は腕・頭などの非脚関節は 0 で初期化
        target_qpos = np.zeros(self.num_motors, dtype=np.float32)
        
        # 右脚 (0-5): 右足のIK結果を使用
        target_qpos[0:6] = qpos_right[0:6]
        
        # 左脚 (6-11): 左足のIK結果を使用
        target_qpos[6:12] = qpos_left[6:12]
        
        # 体幹 (12-13): ニュートラル
        target_qpos[12:14] = 0.0
        
        # 頭部 (14-15): ニュートラル
        target_qpos[14:16] = 0.0
        
        # 右腕 (16-22): ニュートラル（垂直に下ろす）
        target_qpos[16:23] = 0.0
        
        # 左腕 (23-29): ニュートラル（垂直に下ろす）
        target_qpos[23:30] = 0.0
        
        return target_qpos
    
    def step_simulation(self):
        """シミュレーションを1ステップ進める
        
        SteppingController の出力に従って、逐次的に足を制御する。
        """
        try:
            if hasattr(self, 'stepping_controller') and self.stepping_controller:
                # 1. タイマーの更新
                self.timer.time = self.time
                
                # 2. 目標DCMから実際の重心位置をオイラー積分で進展
                com_dot = (self.centroid.dcm_target - self.com_pos) / self.param.T
                self.com_pos += com_dot * self.dt
                
                # 積分結果から現在のDCM参照値をフィードバック
                self.centroid.dcm_ref = self.com_pos.copy()
                self.centroid.zmp_ref = self.centroid.zmp_target.copy()
                
                # 3. 軌道更新 (歩行計画が残っている間だけ実行)
                if len(self.footstep.steps) > 0:
                    self.stepping_controller.update(
                        self.timer, 
                        self.param, 
                        self.footstep,
                        self.footstep_buffer,
                        self.centroid,
                        self.base,
                        self.feet
                    )
                
                # 4. 逆運動学(Genesis IK)の計算と指令
                #    公式のAPIに準拠した形で実施
                self._update_joint_targets_from_feet()
            
        except Exception as e:
            print(f"Warning in step_simulation: {e}")
        
        self.scene.step()
        self.time += self.dt
    


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
    
    def plot_walking_trajectory(self, save_path="walking_trajectory.png"):
        """歩行軌跡を図で出力"""
        if len(self.time_history) == 0:
            print("Warning: No data to plot")
            return
        
        try:
            # NumPy配列に変換
            time_arr = np.array(self.time_history)
            com_arr = np.array(self.com_history)
            zmp_arr = np.array(self.zmp_history)
            lfoot_arr = np.array(self.lfoot_history)
            rfoot_arr = np.array(self.rfoot_history)
            dcm_arr = np.array(self.dcm_history)
            
            # 図の作成
            fig = plt.figure(figsize=(16, 12))
            
            # ===== 1. 俯瞰図（XY平面）=====
            ax1 = plt.subplot(2, 3, 1)
            ax1.plot(com_arr[:, 0], com_arr[:, 1], 'b-', label='CoM', linewidth=2)
            ax1.plot(zmp_arr[:, 0], zmp_arr[:, 1], 'r--', label='ZMP', linewidth=1.5)
            ax1.plot(lfoot_arr[:, 0], lfoot_arr[:, 1], 'g-', label='Left Foot', linewidth=1)
            ax1.plot(rfoot_arr[:, 0], rfoot_arr[:, 1], 'm-', label='Right Foot', linewidth=1)
            
            # ステップの位置をマーク
            if self.footstep:
                for step in self.footstep.steps:
                    ax1.plot(step.zmp[0], step.zmp[1], 'ro', markersize=6)
            
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.set_title('Walking Trajectory (Top View)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # ===== 2. 時系列：X位置 =====
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(time_arr, com_arr[:, 0], 'b-', label='CoM X', linewidth=2)
            ax2.plot(time_arr, zmp_arr[:, 0], 'r--', label='ZMP X', linewidth=1.5)
            ax2.plot(time_arr, lfoot_arr[:, 0], 'g-', label='Left Foot X', alpha=0.7)
            ax2.plot(time_arr, rfoot_arr[:, 0], 'm-', label='Right Foot X', alpha=0.7)
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('X Position [m]')
            ax2.set_title('X Position Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # ===== 3. 時系列：Y位置 =====
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(time_arr, com_arr[:, 1], 'b-', label='CoM Y', linewidth=2)
            ax3.plot(time_arr, zmp_arr[:, 1], 'r--', label='ZMP Y', linewidth=1.5)
            ax3.plot(time_arr, lfoot_arr[:, 1], 'g-', label='Left Foot Y', alpha=0.7)
            ax3.plot(time_arr, rfoot_arr[:, 1], 'm-', label='Right Foot Y', alpha=0.7)
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Y Position [m]')
            ax3.set_title('Y Position Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # ===== 4. 時系列：Z位置（高さ） =====
            ax4 = plt.subplot(2, 3, 4)
            ax4.plot(time_arr, com_arr[:, 2], 'b-', label='CoM Z', linewidth=2)
            ax4.plot(time_arr, lfoot_arr[:, 2], 'g-', label='Left Foot Z', alpha=0.7)
            ax4.plot(time_arr, rfoot_arr[:, 2], 'm-', label='Right Foot Z', alpha=0.7)
            ax4.set_xlabel('Time [s]')
            ax4.set_ylabel('Z Position [m]')
            ax4.set_title('Height Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # ===== 5. DCM軌跡（XY平面） =====
            ax5 = plt.subplot(2, 3, 5)
            ax5.plot(dcm_arr[:, 0], dcm_arr[:, 1], 'c-', label='DCM', linewidth=2)
            ax5.plot(com_arr[:, 0], com_arr[:, 1], 'b--', label='CoM', alpha=0.5)
            ax5.plot(zmp_arr[:, 0], zmp_arr[:, 1], 'r-', label='ZMP', linewidth=1.5)
            ax5.set_xlabel('X [m]')
            ax5.set_ylabel('Y [m]')
            ax5.set_title('DCM Trajectory')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.axis('equal')
            
            # ===== 6. 足の高さ比較 =====
            ax6 = plt.subplot(2, 3, 6)
            ax6.plot(time_arr, lfoot_arr[:, 2], 'g-', label='Left Foot Z', linewidth=2)
            ax6.plot(time_arr, rfoot_arr[:, 2], 'm-', label='Right Foot Z', linewidth=2)
            ax6.fill_between(time_arr, lfoot_arr[:, 2], rfoot_arr[:, 2], alpha=0.2)
            ax6.set_xlabel('Time [s]')
            ax6.set_ylabel('Z Height [m]')
            ax6.set_title('Foot Height Comparison')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Trajectory plot saved to {save_path}")
            
        except Exception as e:
            print(f"Error in plot_walking_trajectory: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close(fig)
    
    def plot_footstep_plan(self, save_path="footstep_plan.png"):
        """改善版：歩行計画（フットステップ）を図で出力"""
        if self.footstep is None or len(self.footstep.steps) == 0:
            print("Warning: No footstep plan to plot")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            steps = self.footstep.steps
            
            # ===== ステップの描画 =====
            colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
            
            for i, step in enumerate(steps):
                foot_width = 0.08
                foot_length = 0.18
                
                # 左足と右足を区別
                if i % 2 == 0:
                    # 左足（緑）
                    edgecolor = 'darkgreen'
                    facecolor_alpha = 0.6
                    label_prefix = 'L'
                else:
                    # 右足（紫）
                    edgecolor = 'darkmagenta'
                    facecolor_alpha = 0.6
                    label_prefix = 'R'
                
                rect = patches.Rectangle(
                    (step.zmp[0] - foot_length/2, step.zmp[1] - foot_width/2),
                    foot_length, foot_width,
                    linewidth=2.5, edgecolor=edgecolor, 
                    facecolor=colors[i], alpha=facecolor_alpha
                )
                ax.add_patch(rect)
                
                # ステップ番号ラベル
                ax.text(step.zmp[0], step.zmp[1], f'{label_prefix}{i}', 
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       color='white')
                
                # ZMP位置を点でプロット
                ax.plot(step.zmp[0], step.zmp[1], 'ko', markersize=5, zorder=10)
                
                # ステップの詳細情報をラベルとして追加
                info_y = step.zmp[1] - foot_width/2 - 0.03
                ax.text(step.zmp[0], info_y, f"x={step.zmp[0]:.3f}m", 
                       ha='center', va='top', fontsize=8, style='italic')
            
            # ===== 矢印で歩行方向を示す =====
            for i in range(len(steps) - 1):
                step_curr = steps[i]
                step_next = steps[i + 1]
                
                # 矢印の線
                ax.annotate('', xy=(step_next.zmp[0], step_next.zmp[1]),
                           xytext=(step_curr.zmp[0], step_curr.zmp[1]),
                           arrowprops=dict(arrowstyle='->', lw=2.5, 
                                         color='black', alpha=0.7, 
                                         connectionstyle="arc3,rad=0.1"))
                
                # ステップ番号を矢印の中点に追加
                mid_x = (step_curr.zmp[0] + step_next.zmp[0]) / 2
                mid_y = (step_curr.zmp[1] + step_next.zmp[1]) / 2 + 0.05
                ax.text(mid_x, mid_y, f'{i}→{i+1}', 
                       ha='center', va='bottom', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # ===== 軸設定 =====
            ax.set_xlabel('X Position [m]', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y Position [m]', fontsize=12, fontweight='bold')
            ax.set_title('Planned Footstep Sequence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axis('equal')
            
            # ===== 凡例 =====
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                      markersize=12, label='Left Foot', markeredgecolor='darkgreen', markeredgewidth=2),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='magenta', 
                      markersize=12, label='Right Foot', markeredgecolor='darkmagenta', markeredgewidth=2),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
            
            # ===== タイトル情報 =====
            info_str = f"Total Steps: {len(steps)} | Stride: {self.param.T:.3f}s | CoM Height: {self.param.com_height:.3f}m"
            ax.text(0.5, -0.1, info_str, transform=ax.transAxes, 
                   ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Footstep plan saved to {save_path}")
            
        except Exception as e:
            print(f"Error in plot_footstep_plan: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close(fig)
    
    def plot_planned_footstep_details(self, save_path="footstep_details.png"):
        """計画されたフットステップの詳細情報を図で出力"""
        if self.footstep is None or len(self.footstep.steps) == 0:
            print("Warning: No footstep plan to plot")
            return
        
        try:
            fig = plt.figure(figsize=(18, 12))
            
            steps = self.footstep.steps
            num_steps = len(steps)
            
            # ===== 1. XY平面上のフットステップと歩行軌跡 =====
            ax1 = plt.subplot(2, 3, 1)
            
            colors = plt.cm.viridis(np.linspace(0, 1, num_steps))
            
            for i, step in enumerate(steps):
                foot_width = 0.08
                foot_length = 0.18
                
                if i % 2 == 0:
                    rect = patches.Rectangle(
                        (step.zmp[0] - foot_length/2, step.zmp[1] - foot_width/2),
                        foot_length, foot_width,
                        linewidth=2, edgecolor='darkgreen', 
                        facecolor=colors[i], alpha=0.7
                    )
                    label_text = f'L{i}'
                else:
                    rect = patches.Rectangle(
                        (step.zmp[0] - foot_length/2, step.zmp[1] - foot_width/2),
                        foot_length, foot_width,
                        linewidth=2, edgecolor='darkmagenta', 
                        facecolor=colors[i], alpha=0.7
                    )
                    label_text = f'R{i}'
                
                ax1.add_patch(rect)
                ax1.text(step.zmp[0], step.zmp[1], label_text, 
                        ha='center', va='center', fontsize=9, fontweight='bold')
                
                if i < num_steps - 1:
                    next_step = steps[i + 1]
                    ax1.annotate('', xy=(next_step.zmp[0], next_step.zmp[1]),
                               xytext=(step.zmp[0], step.zmp[1]),
                               arrowprops=dict(arrowstyle='->', lw=2, 
                                             color='black', alpha=0.6))
            
            ax1.set_xlabel('X Position [m]', fontsize=11)
            ax1.set_ylabel('Y Position [m]', fontsize=11)
            ax1.set_title('Planned Footstep Sequence (Top View)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # ===== 2. ステップインデックス vs X位置 =====
            ax2 = plt.subplot(2, 3, 2)
            
            x_data = [step.zmp[0] for step in steps]
            step_indices = np.arange(num_steps)
            step_labels = [f'L{i}' if i % 2 == 0 else f'R{i}' for i in range(num_steps)]
            
            ax2.plot(step_indices, x_data, 'o-', linewidth=2, markersize=8, color='blue')
            ax2.fill_between(step_indices, x_data, alpha=0.3, color='blue')
            ax2.set_xlabel('Step Index', fontsize=11)
            ax2.set_ylabel('X Position [m]', fontsize=11)
            ax2.set_title('X Position per Step', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(step_indices)
            ax2.set_xticklabels(step_labels, rotation=45)
            
            # ===== 3. ステップインデックス vs Y位置 =====
            ax3 = plt.subplot(2, 3, 3)
            
            y_data = [step.zmp[1] for step in steps]
            ax3.plot(step_indices, y_data, 's-', linewidth=2, markersize=8, color='green')
            ax3.fill_between(step_indices, y_data, alpha=0.3, color='green')
            ax3.set_xlabel('Step Index', fontsize=11)
            ax3.set_ylabel('Y Position [m]', fontsize=11)
            ax3.set_title('Y Position per Step (Foot Spacing)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(step_indices)
            ax3.set_xticklabels(step_labels, rotation=45)
            
            # ===== 4. ZMP位置の3D表現 =====
            ax4 = plt.subplot(2, 3, 4)
            
            zmp_dist = np.linalg.norm(np.array([[step.zmp[0], step.zmp[1]] for step in steps]), axis=1)
            ax4.bar(step_indices, zmp_dist, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax4.set_xlabel('Step Index', fontsize=11)
            ax4.set_ylabel('Distance from Origin [m]', fontsize=11)
            ax4.set_title('ZMP Distance from Origin', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_xticks(step_indices)
            ax4.set_xticklabels(step_labels, rotation=45)
            
            # ===== 5. DCM参照軌跡 =====
            ax5 = plt.subplot(2, 3, 5)
            
            dcm_x = [step.dcm[0] for step in steps]
            dcm_y = [step.dcm[1] for step in steps]
            zmp_x = [step.zmp[0] for step in steps]
            zmp_y = [step.zmp[1] for step in steps]
            
            ax5.plot(dcm_x, dcm_y, 'o-', linewidth=2.5, markersize=10, 
                    label='DCM Reference', color='cyan')
            ax5.plot(zmp_x, zmp_y, 's-', linewidth=2.5, markersize=8, 
                    label='ZMP Reference', color='red')
            
            # DCM-ZMP ベクトル表示
            for i in range(0, num_steps, max(1, num_steps//5)):
                dx = dcm_x[i] - zmp_x[i]
                dy = dcm_y[i] - zmp_y[i]
                ax5.arrow(zmp_x[i], zmp_y[i], dx, dy, 
                         head_width=0.02, head_length=0.02, 
                         fc='orange', ec='orange', alpha=0.6)
            
            ax5.set_xlabel('X [m]', fontsize=11)
            ax5.set_ylabel('Y [m]', fontsize=11)
            ax5.set_title('DCM vs ZMP Reference Trajectory', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=10)
            ax5.grid(True, alpha=0.3)
            ax5.axis('equal')
            
            # ===== 6. ステップ情報テーブル =====
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            table_data = []
            table_data.append(['Step', 'Foot', 'X [m]', 'Y [m]', 'Stepping'])
            
            for i, step in enumerate(steps):
                foot_name = 'Left' if i % 2 == 0 else 'Right'
                stepping = 'Yes' if hasattr(step, 'stepping') and step.stepping else 'No'
                table_data.append([
                    f'{i}',
                    foot_name,
                    f'{step.zmp[0]:.3f}',
                    f'{step.zmp[1]:.3f}',
                    stepping
                ])
            
            table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.1, 0.15, 0.15, 0.15, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # ヘッダー行のスタイル
            for i in range(5):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 交互の行色
            for i in range(1, len(table_data)):
                for j in range(5):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
                    else:
                        table[(i, j)].set_facecolor('white')
            
            ax6.set_title('Footstep Planning Data', fontsize=12, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Footstep details plot saved to {save_path}")
            
        except Exception as e:
            print(f"Error in plot_planned_footstep_details: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close(fig)
    
    def plot_planning_statistics(self, save_path="planning_statistics.png"):
        """計画統計情報を表示"""
        if self.footstep is None or self.param is None:
            print("Warning: No footstep plan to analyze")
            return
        
        try:
            fig = plt.figure(figsize=(16, 10))
            
            steps = self.footstep.steps
            num_steps = len(steps)
            
            if num_steps < 2:
                print("Warning: Not enough steps to analyze")
                return
            
            # ===== 1. ストライド分析 =====
            ax1 = plt.subplot(2, 3, 1)
            
            strides_x = []
            for i in range(1, num_steps, 2):
                if i < num_steps:
                    stride_x = steps[i].zmp[0] - steps[i-1].zmp[0]
                    strides_x.append(stride_x)
            
            if strides_x:
                ax1.bar(np.arange(len(strides_x)), strides_x, alpha=0.7, color='blue')
                ax1.axhline(y=np.mean(strides_x), color='blue', linestyle='--', linewidth=2, 
                           label=f'Mean: {np.mean(strides_x):.3f}m')
            
            ax1.set_xlabel('Stride Number', fontsize=10)
            ax1.set_ylabel('Stride Distance [m]', fontsize=10)
            ax1.set_title('Stride Length Analysis (X Direction)', fontsize=11, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # ===== 2. 足間隔分析 =====
            ax2 = plt.subplot(2, 3, 2)
            
            spacings = [abs(step.zmp[1]) for step in steps]
            
            ax2.plot(spacings, 'o-', linewidth=2, markersize=8, color='green')
            ax2.axhline(y=np.mean(spacings), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(spacings):.3f}m')
            ax2.fill_between(range(len(spacings)), spacings, alpha=0.3, color='green')
            ax2.set_xlabel('Step Index', fontsize=10)
            ax2.set_ylabel('Foot Spacing [m]', fontsize=10)
            ax2.set_title('Foot Spacing Over Steps', fontsize=11, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # ===== 3. ステップ間距離 =====
            ax3 = plt.subplot(2, 3, 3)
            
            step_distances = []
            for i in range(num_steps - 1):
                dx = steps[i+1].zmp[0] - steps[i].zmp[0]
                dy = steps[i+1].zmp[1] - steps[i].zmp[1]
                dist = np.sqrt(dx**2 + dy**2)
                step_distances.append(dist)
            
            if step_distances:
                ax3.plot(step_distances, 'o-', linewidth=2, markersize=8, color='purple')
                ax3.axhline(y=np.mean(step_distances), color='purple', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(step_distances):.3f}m')
                ax3.fill_between(range(len(step_distances)), step_distances, alpha=0.3, color='purple')
            
            ax3.set_xlabel('Step Transition Index', fontsize=10)
            ax3.set_ylabel('Distance [m]', fontsize=10)
            ax3.set_title('Step Distance Analysis', fontsize=11, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # ===== 4. DCM-ZMP距離 =====
            ax4 = plt.subplot(2, 3, 4)
            
            dcm_zmp_distances = []
            for step in steps:
                dist = np.linalg.norm(step.dcm[:2] - step.zmp[:2])
                dcm_zmp_distances.append(dist)
            
            ax4.plot(dcm_zmp_distances, 'o-', linewidth=2, markersize=8, color='red')
            ax4.fill_between(range(len(dcm_zmp_distances)), dcm_zmp_distances, alpha=0.3, color='red')
            ax4.axhline(y=np.max(dcm_zmp_distances), color='red', linestyle='--', 
                       label=f'Max: {np.max(dcm_zmp_distances):.3f}m')
            ax4.set_xlabel('Step Index', fontsize=10)
            ax4.set_ylabel('DCM-ZMP Distance [m]', fontsize=10)
            ax4.set_title('DCM Deviation from ZMP', fontsize=11, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # ===== 5. 統計サマリー =====
            ax5 = plt.subplot(2, 3, 5)
            ax5.axis('off')
            
            summary_text = f"""
Planning Statistics Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Steps: {num_steps}

Stride Length (X):
  Mean: {np.mean(strides_x):.4f} m
  Std:  {np.std(strides_x):.4f} m
  Min:  {np.min(strides_x):.4f} m
  Max:  {np.max(strides_x):.4f} m

Foot Spacing (Y):
  Mean: {np.mean(spacings):.4f} m
  Std:  {np.std(spacings):.4f} m

Step Distance:
  Mean: {np.mean(step_distances):.4f} m
  Min:  {np.min(step_distances):.4f} m
  Max:  {np.max(step_distances):.4f} m

DCM-ZMP Distance:
  Mean: {np.mean(dcm_zmp_distances):.4f} m
  Max:  {np.max(dcm_zmp_distances):.4f} m

Parameters:
  CoM Height: {self.param.com_height:.3f} m
  Time Constant: {self.param.T:.3f} s
"""
            
            ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # ===== 6. 速度分析 =====
            ax6 = plt.subplot(2, 3, 6)
            
            velocities = []
            for i in range(len(steps) - 1):
                dx = steps[i+1].zmp[0] - steps[i].zmp[0]
                dy = steps[i+1].zmp[1] - steps[i].zmp[1]
                dist = np.sqrt(dx**2 + dy**2)
                velocities.append(dist)
            
            if velocities:
                ax6.plot(velocities, 'o-', linewidth=2, markersize=8, color='orange')
                ax6.axhline(y=np.mean(velocities), color='orange', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(velocities):.3f} m')
                ax6.fill_between(range(len(velocities)), velocities, alpha=0.3, color='orange')
            
            ax6.set_xlabel('Step Transition Index', fontsize=10)
            ax6.set_ylabel('Distance [m]', fontsize=10)
            ax6.set_title('Step Movement Distance', fontsize=11, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Planning statistics saved to {save_path}")
            
        except Exception as e:
            print(f"Error in plot_planning_statistics: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close(fig)
    
    def plot_stability_analysis(self, save_path="stability_analysis.png"):
        """安定性解析（CoM-ZMP距離など）を図で出力"""
        if len(self.time_history) == 0:
            print("Warning: No data to plot")
            return
        
        try:
            time_arr = np.array(self.time_history)
            com_arr = np.array(self.com_history)
            zmp_arr = np.array(self.zmp_history)
            
            # CoM-ZMP距離を計算
            com_zmp_dist = np.linalg.norm(com_arr[:, :2] - zmp_arr[:, :2], axis=1)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # ===== 1. CoM-ZMP距離 =====
            ax = axes[0, 0]
            ax.plot(time_arr, com_zmp_dist, 'b-', linewidth=2)
            ax.fill_between(time_arr, 0, com_zmp_dist, alpha=0.3)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Distance [m]')
            ax.set_title('CoM-ZMP Distance (Stability Measure)')
            ax.grid(True, alpha=0.3)
            
            # ===== 2. X方向の安定性マージン =====
            ax = axes[0, 1]
            com_x = com_arr[:, 0]
            zmp_x = zmp_arr[:, 0]
            ax.plot(time_arr, com_x - zmp_x, 'r-', linewidth=2, label='CoM X - ZMP X')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.fill_between(time_arr, 0, com_x - zmp_x, alpha=0.3)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Position [m]')
            ax.set_title('X Direction Stability')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # ===== 3. Y方向の安定性マージン =====
            ax = axes[1, 0]
            com_y = com_arr[:, 1]
            zmp_y = zmp_arr[:, 1]
            ax.plot(time_arr, com_y - zmp_y, 'g-', linewidth=2, label='CoM Y - ZMP Y')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.fill_between(time_arr, 0, com_y - zmp_y, alpha=0.3)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Position [m]')
            ax.set_title('Y Direction Stability')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # ===== 4. CoM-ZMP VectorField =====
            ax = axes[1, 1]
            # サンプルポイントを取得
            sample_indices = np.linspace(0, len(time_arr) - 1, 30, dtype=int)
            for idx in sample_indices:
                ax.arrow(zmp_arr[idx, 0], zmp_arr[idx, 1],
                        com_arr[idx, 0] - zmp_arr[idx, 0],
                        com_arr[idx, 1] - zmp_arr[idx, 1],
                        head_width=0.02, head_length=0.02, fc='blue', ec='blue', alpha=0.6)
            
            ax.plot(com_arr[:, 0], com_arr[:, 1], 'b-', alpha=0.3, label='CoM')
            ax.plot(zmp_arr[:, 0], zmp_arr[:, 1], 'r-', alpha=0.3, label='ZMP')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title('CoM-ZMP Vector Field')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Stability analysis saved to {save_path}")
            
        except Exception as e:
            print(f"Error in plot_stability_analysis: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close(fig)
    
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
        
        # 【改善】データ可視化（エラーハンドリング付き）
        print("\nGenerating plots...")
        try:
            self.plot_planned_footstep_details("footstep_details.png")
        except Exception as e:
            print(f"Warning: Could not generate footstep_details: {e}")
        
        try:
            self.plot_planning_statistics("planning_statistics.png")
        except Exception as e:
            print(f"Warning: Could not generate planning_statistics: {e}")
        
        try:
            self.plot_footstep_plan("footstep_plan.png")
        except Exception as e:
            print(f"Warning: Could not generate footstep_plan: {e}")
        
        try:
            self.plot_walking_trajectory("walking_trajectory.png")
        except Exception as e:
            print(f"Warning: Could not generate walking_trajectory: {e}")
        
        try:
            self.plot_stability_analysis("stability_analysis.png")
        except Exception as e:
            print(f"Warning: Could not generate stability_analysis: {e}")
        
        print("✓ All plots completed.")
    


def main():
    parser = argparse.ArgumentParser(description='HRP2 Walking Simulation')
    parser.add_argument('--stride', type=float, default=0.1, help='Stride distance (m)')
    parser.add_argument('--sway', type=float, default=0.0, help='Sway (m)')
    parser.add_argument('--turn', type=float, default=0.0, help='Turn angle (rad)')
    parser.add_argument('--spacing', type=float, default=0.2, help='Foot spacing (m)')
    parser.add_argument('--com-height', type=float, default=0.65, help='CoM height (m)')
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