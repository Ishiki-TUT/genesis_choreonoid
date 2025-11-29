import torch
import math

from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from tensordict import TensorDict

def rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

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
        ###
        self.base_z_range_min = self.env_cfg['base_z_noise'][0] if 'base_z_noise' in self.env_cfg else 0.0
        self.base_z_range_max = self.env_cfg['base_z_noise'][1] if 'base_z_noise' in self.env_cfg else 0.0
        self.base_pitch_range_min = self.env_cfg['base_pitch_noise'][0] if 'base_pitch_noise' in self.env_cfg else 0.0
        self.base_pitch_range_max = self.env_cfg['base_pitch_noise'][1] if 'base_pitch_noise' in self.env_cfg else 0.0
        self.base_roll_range_min = self.env_cfg['base_roll_noise'][0] if 'base_roll_noise' in self.env_cfg else 0.0
        self.base_roll_range_max = self.env_cfg['base_roll_noise'][1] if 'base_roll_noise' in self.env_cfg else 0.0

        # self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # w成分を-1倍して逆クオータニオンを作成
        self.inv_base_init_quat = self.base_init_quat.clone()
        self.inv_base_init_quat[0] *= -1.0

        #### call overrided function
        self.scene_build(substeps, robot_urdf_path, show_viewer)

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
        self.use_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32) ##
        self.use_base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32) ##
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
        ##
        self.reset()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)


    def step(self, actions):
        self.exact_actions = torch.tensor(actions, device=self.device, dtype=torch.float32) ## copy
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        self.target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        self.env_step() ## call overrided function
        self.update_buffers() ## call overrided function

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
            # print(f"reward {name}: {rew.mean().item():.4f}")

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

        self.extras["observations"]["critic"] = self.obs_buf ## to be fixed

        return TensorDict({"policy": self.obs_buf}, device=self.device), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf ## to be fixed
        return TensorDict({"policy": self.obs_buf}, device=self.device)

    def get_privileged_observations(self):
        ###
        return None

    def reset_buffers_idx(self, envs_idx):
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.dof_force[envs_idx] = 0.0

        # reset base
        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        ### randomize
        sz = len(envs_idx)
        self.use_base_pos[envs_idx, 2] = self.base_pos[envs_idx, 2] + rand_float(self.base_z_range_min, self.base_z_range_max, (sz,), self.device)
        buf_xyz = torch.zeros((sz, 3), device=self.device, dtype=torch.float32)
        buf_xyz[:, 0] = rand_float(self.base_roll_range_min,  self.base_roll_range_max,  (sz,), self.device) ## deg
        buf_xyz[:, 1] = rand_float(self.base_pitch_range_min, self.base_pitch_range_max, (sz,), self.device) ## deg
        self.use_base_quat[envs_idx] = transform_quat_by_quat(self.base_quat[envs_idx], xyz_to_quat(buf_xyz))

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0

        # reset buffers
        self.actions[envs_idx] = 0.0 ## ??
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
        ###
        self.reset_buffers_idx(envs_idx)
        self.reset_env_idx(envs_idx) ## call overrided function
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs = self.get_observations()
        return obs, None

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

    def _reward_plus_watt(self):
        return 1 / (1 + torch.sum( torch.clamp(self.dof_force * self.dof_vel, min = 0), dim = 1 ))

    #def _reward_min_ankle_height(self):
    #    # ankle height
    #    return torch.square(self.min_ankle_height)
