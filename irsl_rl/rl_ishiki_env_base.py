import torch
import math

def rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# def quat_from_euler_xyz(roll, pitch, yaw):
#     """オイラー角(roll, pitch, yaw)からクォータニオンを生成"""
#     cr = torch.cos(roll * 0.5)
#     sr = torch.sin(roll * 0.5)
#     cp = torch.cos(pitch * 0.5)
#     sp = torch.sin(pitch * 0.5)
#     cy = torch.cos(yaw * 0.5)
#     sy = torch.sin(yaw * 0.5)
    
#     w = cr * cp * cy + sr * sp * sy
#     x = sr * cp * cy - cr * sp * sy
#     y = cr * sp * cy + sr * cp * sy
#     z = cr * cp * sy - sr * sp * cy
    
#     return torch.stack([w, x, y, z], dim=-1)

class RLEnvBase:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=True, device="cuda",
                 dt=0.02, substeps=2, robot_urdf_path=None):

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = device

        self.simulate_action_latency = True
        self.dt = dt
        self.substeps = substeps
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)

        # self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # w成分を-1倍して逆クオータニオンを作成
        self.inv_base_init_quat = self.base_init_quat.clone()
        self.inv_base_init_quat[0] *= -1.0

        # ランダム化用のバッファを追加
        self.randomized_kp = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.randomized_kd = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.randomized_base_height = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        # **重要**: scene_build()の前に初期化
        self._initialize_randomization()

        #### call
        self.scene_build(substeps, robot_urdf_path, show_viewer)
        
        # scene_build()後にパラメータを適用
        if hasattr(self, '_update_all_robot_parameters'):
            self._update_all_robot_parameters()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.int32)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=torch.float32)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=torch.float32,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dof_force = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.last_dof_force = torch.zeros_like(self.actions)
        self.target_dof_pos = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=torch.float32,
        )
        #self.l_ankle_z = torch.zeros((self.num_envs), device=self.device, dtype=torch.float32)
        #self.r_ankle_z = torch.zeros((self.num_envs), device=self.device, dtype=torch.float32)
        #self.min_ankle_height = torch.zeros((self.num_envs), device=self.device, dtype=torch.float32)
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        # ランダム化用のバッファを追加
        self.randomized_kp = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.randomized_kd = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.randomized_base_height = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        # **重要**: scene_build()の前に初期化
        self._initialize_randomization()


    def _initialize_randomization(self):
        """ランダム化パラメータの初期化"""
        if "randomization" in self.env_cfg:
            rand_cfg = self.env_cfg["randomization"]
            
            # kp, kdの初期化
            if rand_cfg.get("randomize_kp_kd", True):
                for env_idx in range(self.num_envs):
                    self.randomized_kp[env_idx] = rand_float(
                        rand_cfg["kp_range"][0], rand_cfg["kp_range"][1], 
                        (self.num_actions,), self.device
                    )
                    self.randomized_kd[env_idx] = rand_float(
                        rand_cfg["kd_range"][0], rand_cfg["kd_range"][1], 
                        (self.num_actions,), self.device
                    )
            else:
                self.randomized_kp.fill_(self.env_cfg["kp"])
                self.randomized_kd.fill_(self.env_cfg["kd"])
            
            # 初期高さの初期化
            if rand_cfg.get("randomize_height", True):
                self.randomized_base_height[:] = rand_float(
                    rand_cfg["base_height_range"][0], rand_cfg["base_height_range"][1],
                    (self.num_envs,), self.device
                )
            else:
                self.randomized_base_height.fill_(self.env_cfg["base_init_pos"][2])
        else:
            # ランダム化なしの場合
            self.randomized_kp.fill_(self.env_cfg["kp"])
            self.randomized_kd.fill_(self.env_cfg["kd"])
            self.randomized_base_height.fill_(self.env_cfg["base_init_pos"][2])

    def _randomize_env_params(self, envs_idx):
        """指定された環境のパラメータをランダム化"""
        if "randomization" not in self.env_cfg:
            return
            
        rand_cfg = self.env_cfg["randomization"]
        
        if not rand_cfg.get("randomize_every_reset", True):
            return
            
        num_reset_envs = len(envs_idx)
        if num_reset_envs == 0:
            return
            
        # kp, kdのランダム化
        if rand_cfg.get("randomize_kp_kd", True):
            for i, env_idx in enumerate(envs_idx):
                self.randomized_kp[env_idx] = rand_float(
                    rand_cfg["kp_range"][0], rand_cfg["kp_range"][1], 
                    (self.num_actions,), self.device
                )
                self.randomized_kd[env_idx] = rand_float(
                    rand_cfg["kd_range"][0], rand_cfg["kd_range"][1], 
                    (self.num_actions,), self.device
                )
        
        # 初期高さのランダム化
        if rand_cfg.get("randomize_height", True):
            self.randomized_base_height[envs_idx] = rand_float(
                rand_cfg["base_height_range"][0], rand_cfg["base_height_range"][1],
                (num_reset_envs,), self.device
            )

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)


    def step(self, actions):
        self.exact_actions = torch.tensor(actions, device=self.device, dtype=torch.float32) ## copy
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        self.target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        self.env_step()
        self.update_buffers()

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float32)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_dof_force[:] = self.dof_force[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        ###
        return None

    def reset_buffers_idx(self, envs_idx):
        # パラメータをランダム化
        self._randomize_env_params(envs_idx)
        
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.dof_force[envs_idx] = 0.0

        num_reset_envs = len(envs_idx)
        if num_reset_envs > 0:
            # ベース位置（高さをランダム化）
            base_pos_randomized = self.base_init_pos.clone().unsqueeze(0).repeat(num_reset_envs, 1)
            base_pos_randomized[:, 2] = self.randomized_base_height[envs_idx]
            self.base_pos[envs_idx] = base_pos_randomized
            
            # ランダムな姿勢（既存のコード）
            # random_roll = rand_float(-0.5, 0.5, (num_reset_envs,), self.device) * (math.pi / 180.0)
            random_roll = torch.deg2rad(rand_float(-0.5, 0.5, (num_reset_envs,), self.device))
            # random_pitch = rand_float(-0.5, 0.5, (num_reset_envs,), self.device) * (math.pi / 180.0)
            random_pitch = torch.deg2rad(rand_float(-0.5, 0.5, (num_reset_envs,), self.device))
            zero_yaw = torch.zeros((num_reset_envs,), device=self.device, dtype=torch.float32)
            
            # ランダムなクォータニオンを計算して設定
            random_quat = (random_roll, random_pitch, zero_yaw)
            self.base_quat[envs_idx] = random_quat

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.reset_buffers_idx(envs_idx)
        self.reset_env_idx(envs_idx)
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def zeoAction(self):
        return torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)

    # ------------ abstract functions ----------------
    def scene_build(self, substeps, robot_urdf_path, show_viewer, max_effort, pre_wait):
        raise NotImplementedError()

    def env_step(self):
        raise NotImplementedError()

    def update_buffers(self):
        raise NotImplementedError()

    def reset_env_idx(self, envs_idx):
        raise NotImplementedError()

    # ------------ reward functions ----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_effort(self):
        # Penalize effort
        #return torch.sum(torch.abs(self.dof_force), dim=1)
        return torch.sum(torch.square(self.dof_force), dim=1)

    def _reward_base_rotation_P(self):
        # Penalize base rotation
        return torch.abs(self.base_euler[:, 1])

    def _reward_base_rotation_R(self):
        # Penalize base rotation
        return torch.abs(self.base_euler[:, 0])

    def _reward_episode_len(self):
        return torch.ones((self.num_envs,), dtype=torch.float32)

    def _reward_correct_action(self):
        return 1 / (1 + torch.sum(torch.square(self.exact_actions - self.actions), dim=1))

    def _reward_joint_position_error(self):
        return 1 / (1 + torch.sum(torch.abs(self.target_dof_pos - self.dof_pos), dim=1))

    #def _reward_min_ankle_height(self):
    #    # ankle height
    #    return torch.square(self.min_ankle_height)
