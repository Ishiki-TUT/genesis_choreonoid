import torch
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from rl_env_base import RLEnvBase

class RLEnvGenesis(RLEnvBase):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True, device="cuda",
                 dt=0.02, substeps=2, robot_urdf_path=None, **kwargs):
        ##
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, device, dt, substeps, robot_urdf_path)

    def build_environment(self):
        # add plane
        # self.plane = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.plane = self.scene.add_entity(
            morph=gs.morphs.Terrain(
                n_subterrains=(7, 7),
                subterrain_size=(3.0, 3.0),
                horizontal_scale=0.25,
                vertical_scale=0.005, # 0.005
                pos=(-7.5, -10.5, 0.0),
                subterrain_types=[
                    ["fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain","fractal_terrain","fractal_terrain"],
                    ["fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain","fractal_terrain","fractal_terrain"],
                    ["fractal_terrain", "fractal_terrain", "fractal_terrain", "flat_terrain", "fractal_terrain", "fractal_terrain","fractal_terrain"],
                    ["fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain","fractal_terrain","fractal_terrain"],
                    ["fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain","fractal_terrain","fractal_terrain"],
                    ["fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain","fractal_terrain","fractal_terrain"],
                    ["fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain", "fractal_terrain","fractal_terrain","fractal_terrain"],
                ],
            ),
        )

    def scene_build(self, substeps, robot_urdf_path, show_viewer):
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            # vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        ### build environments geometries
        self.build_environment()

        # add robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=robot_urdf_path,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                fixed=False # ロボットを固定
            ),
        )

        # build
        self.scene.build(n_envs=self.num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        #if max_effort is not None:
        #    self.robot.set_dofs_force_range([-max_effort]*self.num_actions, [max_effort]*self.num_actions, self.motors_dof_idx)

    def randomize_domain_parameters(self, envs_idx):

        if "domain_rand" not in self.env_cfg:
            return

        dr = self.env_cfg["domain_rand"]

        fr_low, fr_high = dr["friction"]
        res_low, res_high = dr["restitution"]

        fric = float(torch.rand(1, device=self.device) * (fr_high - fr_low) + fr_low)
        rest = float(torch.rand(1, device=self.device) * (res_high - res_low) + res_low)

        # --- 地面（plane）の geom に設定 ---
        try:
            self.plane.set_friction(fric)
            print(" [DomainRand] Set ground friction to:", fric)
            # self.plane.set_restitution(rest)
        except Exception as e:
            print(f"[DomainRand] Warning: ground friction/restitution not set: {e}")

        # --- PDゲイン ---
        dr = self.env_cfg["domain_rand"]
        kp_range = dr["kp"]
        kd_range = dr["kd"]
        
        # リセットされる環境1つずつに対して設定を行う（確実な方法）
        # ※Pythonループは遅いですが、リセット時は許容範囲です
        for i, env_id in enumerate(envs_idx):
            # ランダムなゲインを生成
            kp = np.random.uniform(kp_range[0], kp_range[1])
            kd = np.random.uniform(kd_range[0], kd_range[1])
            
            # その環境の全関節に適用
            # envs_idx=[env_id] とすることで、その環境だけ更新
            self.robot.set_dofs_kp([kp]*self.num_actions, self.motors_dof_idx, envs_idx=[env_id])
            self.robot.set_dofs_kv([kd]*self.num_actions, self.motors_dof_idx, envs_idx=[env_id])


    def env_step(self): ## override
        self.robot.control_dofs_position(self.target_dof_pos, self.motors_dof_idx)
        self.scene.step()


    def update_buffers(self): ## override
        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.dof_force[:] = self.robot.get_dofs_force(self.motors_dof_idx)
        # self.l_ankle_z[:] = self.robot.get_link(name="L_ANKLE_R").get_pos()[:,2]
        # self.r_ankle_z[:] = self.robot.get_link(name="R_ANKLE_R").get_pos()[:,2]
        # self.min_ankle_height[:], _ = torch.min(torch.stack([self.l_ankle_z,self.r_ankle_z]), dim=0)


    def reset_env_idx(self, envs_idx): ## override
        self.robot.set_dofs_position(
            position = self.dof_pos[envs_idx],
            dofs_idx_local = self.motors_dof_idx,
            zero_velocity  = True,
            envs_idx = envs_idx,
        )

        self.robot.set_pos (self.use_base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.use_base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        self.robot.zero_all_dofs_velocity(envs_idx)

        # --- Domain Randomization の適用 ---
        if hasattr(self, "randomize_domain_parameters"):
            self.randomize_domain_parameters(envs_idx)
