import torch
import numpy as np
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, xyz_to_quat

from rl_env_base import RLEnvBase

class RLEnvGenesis(RLEnvBase):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True, device="cuda",
                 dt=0.02, substeps=2, robot_urdf_path=None, **kwargs):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, device, dt, substeps, robot_urdf_path)
        
        self.randomize_terrain_properties()

    def build_environment(self):
        self.floor_entities = []
        
        # --- 設定 ---
        self.num_terrains = 8
        self.terrain_size = (50.0, 50.0)
        self.spacing = 1.0
        # ------------

        side = math.ceil(math.sqrt(self.num_terrains))
        self.terrain_grid = (side, side)
        self.box_centers_cpu = []

        total_width = self.terrain_grid[1] * self.terrain_size[0] + (self.terrain_grid[1] - 1) * self.spacing
        total_height = self.terrain_grid[0] * self.terrain_size[1] + (self.terrain_grid[0] - 1) * self.spacing
        
        start_x = -total_width / 2.0 + self.terrain_size[0] / 2.0
        start_y = -total_height / 2.0 + self.terrain_size[1] / 2.0

        for i in range(self.num_terrains):
            row = i // self.terrain_grid[1]
            col = i % self.terrain_grid[1]
            
            pos_x_global = start_x + col * (self.terrain_size[0] + self.spacing)
            pos_y_global = start_y + row * (self.terrain_size[1] + self.spacing)
            
            self.box_centers_cpu.append([pos_x_global, pos_y_global]) 
            
            floor_entity = self.scene.add_entity(
                gs.morphs.Box(
                    size=(self.terrain_size[0], self.terrain_size[1], 1.0),
                    pos=(pos_x_global, pos_y_global, -0.5),
                    fixed=True
                ),
                material=gs.materials.Rigid(friction=1.0)
            )
            self.floor_entities.append(floor_entity)
        
        print(f"[Terrain] Created {self.num_terrains} terrains.")

    def _calculate_base_init_pos_from_boxes(self):
        self.terrain_indices = torch.arange(self.num_envs, device=self.device) % self.num_terrains
        
        original_pos = self.base_init_pos.cpu().numpy()
        if original_pos.ndim == 1:
            base_pos_cpu = np.tile(original_pos, (self.num_envs, 1))
        else:
            base_pos_cpu = original_pos.copy()

        if not hasattr(self, 'box_centers_cpu') or not self.box_centers_cpu:
             box_centers_arr = np.zeros((self.num_terrains, 2))
        else:
             box_centers_arr = np.array(self.box_centers_cpu)
             if box_centers_arr.ndim == 1:
                 box_centers_arr = box_centers_arr.reshape(-1, 2)

        terrain_indices_cpu = self.terrain_indices.cpu().numpy()
        
        base_pos_cpu[:, 0] = box_centers_arr[terrain_indices_cpu, 0]
        base_pos_cpu[:, 1] = box_centers_arr[terrain_indices_cpu, 1]
        base_pos_cpu[:, 2] = 0.6  
        
        base_pos_cpu = base_pos_cpu.astype(np.float32)
        self.base_init_pos = torch.from_numpy(base_pos_cpu).to(self.device)

        print(f"[InitPos] Env 0 assigned to Box {terrain_indices_cpu[0]}. Spawn Pos: {base_pos_cpu[0]}")

    def scene_build(self, substeps, robot_urdf_path, show_viewer):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(64, self.num_envs)))), 
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.build_environment()
        self._calculate_base_init_pos_from_boxes()

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=robot_urdf_path,
                pos=(0.0, 0.0, 1.0),
                quat=(1.0, 0.0, 0.0, 0.0),
                fixed=False
            ),
        )

        self.scene.build(n_envs=self.num_envs)

        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

    def randomize_terrain_properties(self):
        if "domain_rand" not in self.env_cfg:
            return
        dr = self.env_cfg["domain_rand"]
        fr_low, fr_high = dr.get("friction", [0.5, 1.0])
        fric_values = torch.linspace(fr_low, fr_high, self.num_terrains, device=self.device)
        try:
            print("-" * 50)
            print(f" [Terrain] Configuring {self.num_terrains} terrains (Range: {fr_low} - {fr_high})")
            for i in range(self.num_terrains):
                if i < len(self.floor_entities):
                    val = fric_values[i].item()
                    self.floor_entities[i].set_friction(val)
                    print(f"   | Terrain {i}: Friction = {val:.4f}")
            print("-" * 50)
        except Exception as e:
            print(f"[Terrain] Warning: Could not set friction: {e}")

    def env_step(self): 
        self.robot.control_dofs_position(self.target_dof_pos, self.motors_dof_idx)
        self.scene.step()

    def update_buffers(self): 
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

    # 【重要】親クラスの機能をマルチ地形用に適合させて再実装
    def reset_buffers_idx(self, envs_idx):
        # 1. 関節のリセット
        if hasattr(self, 'default_dof_pos'):
            self.dof_pos[envs_idx] = self.default_dof_pos
        else:
            self.dof_pos[envs_idx] = 0.0
        self.dof_vel[envs_idx] = 0.0
        self.dof_force[envs_idx] = 0.0

        # 2. ベース位置・姿勢のリセット (マルチ地形対応)
        # 親クラスは一括代入ですが、ここはインデックス指定が必要です
        self.base_pos[envs_idx] = self.base_init_pos[envs_idx]
        
        if self.base_init_quat.ndim == 2:
            self.base_quat[envs_idx] = self.base_init_quat[envs_idx]
        else:
            self.base_quat[envs_idx] = self.base_init_quat

        # 3. ランダマイゼーション (親クラスのロジックを復元)
        # ここで use_base_pos / use_base_quat を生成します
        sz = len(envs_idx)
        
        # 高さ(Z)のノイズ付加
        z_noise = (self.base_z_range_max - self.base_z_range_min) * torch.rand((sz,), device=self.device) + self.base_z_range_min
        
        # 重要: use_base_pos に現在のBox位置(base_pos)をコピーしてからZノイズを足す
        self.use_base_pos[envs_idx] = self.base_pos[envs_idx].clone()
        self.use_base_pos[envs_idx, 2] += z_noise

        # 姿勢(Roll/Pitch)のノイズ付加
        roll_noise = (self.base_roll_range_max - self.base_roll_range_min) * torch.rand((sz,), device=self.device) + self.base_roll_range_min
        pitch_noise = (self.base_pitch_range_max - self.base_pitch_range_min) * torch.rand((sz,), device=self.device) + self.base_pitch_range_min
        
        buf_xyz = torch.zeros((sz, 3), device=self.device, dtype=torch.float32)
        buf_xyz[:, 0] = roll_noise # Roll
        buf_xyz[:, 1] = pitch_noise # Pitch
        # Yawはそのまま
        
        # ノイズを加えたクォータニオンを計算
        self.use_base_quat[envs_idx] = transform_quat_by_quat(self.base_quat[envs_idx], xyz_to_quat(buf_xyz))

        # 4. その他バッファのリセット
        self.base_lin_vel[envs_idx] = 0.
        self.base_ang_vel[envs_idx] = 0.
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # 5. Extras (ログ用)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset_env_idx(self, envs_idx): 
        self.robot.set_dofs_position(
            position = self.dof_pos[envs_idx],
            dofs_idx_local = self.motors_dof_idx,
            zero_velocity  = True,
            envs_idx = envs_idx,
        )
        
        # 【重要】ランダムノイズを加えた use_base_pos / use_base_quat を使用する
        self.robot.set_pos(self.use_base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.use_base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        
        self.robot.zero_all_dofs_velocity(envs_idx)

        if hasattr(self, "randomize_domain_parameters"):
             self.randomize_domain_parameters(envs_idx)