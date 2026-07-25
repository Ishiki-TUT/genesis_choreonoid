import numpy as np
import genesis as gs
import os

# from bex24_env_gs import Bex24Env as RLEnv

gs.init(backend=gs.gpu)

# scene = gs.Scene(show_viewer=True)
dt = 0.01
substeps = 5

scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=dt,
                substeps=substeps,
                gravity=(0.0, 0.0, -9.81),
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            # vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=True,
        )
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
# franka = scene.add_entity(
#     gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
# )

robot_urdf_path = "hrp2_description/HRP2_genesis.urdf"
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # /userdir
# robot_urdf_path = os.path.join(ROOT, "userdir", "humanoid_research_k", "robots", "kawada_base.simple_collision.urdf")

robot = scene.add_entity(
    gs.morphs.URDF(
        file=robot_urdf_path,
        pos = [0, 0, 0.71], # 直立時の足裏までの距離は約0.705m。少し浮かせて0.72mあたりにするのが安全
        euler = (0, 0, 0),
        fixed=False,
    ),
)

scene.build()

# 全身の関節名リスト（Genesis/URDFで定義されている名前）
joint_names = [
    # --- 下半身 (12) ---
    'RLEG_JOINT0', 'RLEG_JOINT1', 'RLEG_JOINT2', 'RLEG_JOINT3', 'RLEG_JOINT4', 'RLEG_JOINT5',
    'LLEG_JOINT0', 'LLEG_JOINT1', 'LLEG_JOINT2', 'LLEG_JOINT3', 'LLEG_JOINT4', 'LLEG_JOINT5',
    # --- 体幹・頭部 (4) ---
    'CHEST_JOINT0', 'CHEST_JOINT1', # Waist Yaw, Waist Pitch
    'HEAD_JOINT0',  'HEAD_JOINT1',  # Head Yaw, Head Pitch
    # --- 右腕 (7) ---
    'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5', 'RARM_JOINT6',
    # --- 左腕 (7) ---
    'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'LARM_JOINT6',
    ]

# 制御に使うDOFインデックス
motors_dof_idx = [robot.get_joint(name).dof_start for name in joint_names]
n_dofs = len(motors_dof_idx) # 30

# 初期関節角（ハーフシッティング＋腕を少し下ろした姿勢）
q_init = np.array([
    # Legs (Right, Left)
    0, 0, -0.4, 0.8, -0.4, 0,
    0, 0, -0.4, 0.8, -0.4, 0,
    # Chest (Yaw, Pitch)
    0.0, 0.0,
    # Head (Yaw, Pitch)
    0.0, 0.0, 
    # R Arm (Shoulder: P, R, Y, Elbow: P, Wrist: Y, P, R)
    0.0, -0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
    # L Arm
    0.0,  0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
], dtype=np.float32)

# 1) PDゲインとトルク範囲の強化
# 全身に関数を適用するため、リストの長さを自動調整しています
robot.set_dofs_kp([2000.0] * n_dofs, motors_dof_idx)
robot.set_dofs_kv([100.0] * n_dofs, motors_dof_idx)
robot.set_dofs_force_range([-1000.0] * n_dofs, [1000.0] * n_dofs, motors_dof_idx)

# 2) 初期姿勢を「直立（ゼロ）」に変更
start_pos = np.zeros(n_dofs, dtype=np.float32)
robot.set_dofs_position(start_pos, motors_dof_idx)
robot.set_dofs_velocity(np.zeros(n_dofs, dtype=np.float32), motors_dof_idx)
# ルートの速度もゼロ（APIがあれば）
# robot.set_root_velocity(lin=(0,0,0), ang=(0,0,0))


# 3) 毎ステップはteleport(set_)ではなくPD制御（control_）で目標を保持

print("Press Enter to start simulation...") # わかりやすくプロンプトを表示
if input() == "":
    # scene.step() # ここで1回だけstepすると一瞬で終わるので削除またはコメントアウト
    # input()
    
    for i in range(1000): # ループ回数
        # ここで目標姿勢(q_init)を指定することで、0 -> q_init へ動こうとする力が発生します
        robot.control_dofs_position(q_init, motors_dof_idx)
        
        scene.step()