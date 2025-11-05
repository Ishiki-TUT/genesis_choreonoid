import numpy as np
import genesis as gs

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

robot_urdf_path = "bp000.urdf"

robot = scene.add_entity(
    gs.morphs.URDF(
        file=robot_urdf_path,
        pos = [0, 0, 0.64],
        euler = (0, 0, 0),
        fixed=False,
    ),
)

scene.build()

joint_names = [
    'R_HIP_Y',
    'R_HIP_R',
    'R_HIP_P',
    'R_KNEE',
    'R_ANKLE_P',
    'R_ANKLE_R',
    'L_HIP_Y',
    'L_HIP_R',
    'L_HIP_P',
    'L_KNEE',
    'L_ANKLE_P',
    'L_ANKLE_R',
    ]

# 制御に使うDOFインデックス（control_*系は dof_start を使うのが安全）
motors_dof_idx = [robot.get_joint(name).dof_start for name in joint_names]

# 初期関節角（足裏が床に近い安定姿勢）
q_init = np.array([0, 0, -0.8, 1.6, -0.8, 0,
                   0, 0, -0.8, 1.6, -0.8, 0], dtype=np.float32)

# 1) PDゲインとトルク範囲（強すぎない値から）
robot.set_dofs_kp([3000.0]*12, motors_dof_idx)
robot.set_dofs_kv([10.0]*12, motors_dof_idx)
robot.set_dofs_force_range([-30000.0]*12, [30000.0]*12, motors_dof_idx)

# 2) 初期姿勢・速度を明示的にゼロ化
robot.set_dofs_position(q_init, motors_dof_idx)
robot.set_dofs_velocity(np.zeros(12, dtype=np.float32), motors_dof_idx)
# ルートの速度もゼロ（APIがあれば）
# robot.set_root_velocity(lin=(0,0,0), ang=(0,0,0))


# 3) 毎ステップはteleport(set_)ではなくPD制御（control_）で目標を保持

if input() == "":
    scene.step()    
    input()
    for i in range(10000):
        robot.control_dofs_position(q_init, motors_dof_idx)
        # robot.set_dofs_position(q_init, motors_dof_idx)
        # input()
        scene.step()