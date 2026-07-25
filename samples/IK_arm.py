import numpy as np
import genesis as gs

########################## 初期化 ##########################
gs.init(backend=gs.gpu)

########################## シーンを作成 ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3, -1, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

########################## エンティティ ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size = (0.04, 0.04, 0.04),
        pos  = (0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
########################## シーンを構築 ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# 制御ゲインを設定
# 注意: 以下の値は Franka に最適化された値です
# ロボットごとに異なるパラメータセットが必要になる場合があります。
# 高品質な URDF または XML ファイルが提供されていると、
# それをパースしてパラメータが自動的に設定される場合もあります。
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)


# エンドエフェクタのリンクを取得
end_effector = franka.get_link('hand')

# 把持前の姿勢へ移動
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.25]),
    quat = np.array([1, 1, 0, 0]),
)
# グリッパーが開いた状態の位置
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2 秒間の移動
)
# 計画された経路を実行
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()

# ロボットが最後のウェイポイントに到達する時間を確保
for i in range(100):
    scene.step()