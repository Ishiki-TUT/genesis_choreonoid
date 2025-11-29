import sys
import os

from rl_env_gs import RLEnvGenesis
# from rl_env_gs_tr import RLEnvGenesis
import genesis as gs
import torch
import numpy as np


def get_link_lowest_point_z(link):
    # 各 geom の AABB の最小コーナー（min_xyz）を取得
    aabb_mins = []
    for g in link.geoms:
        aabb = g.get_AABB()  # shape: (2, 3)
        aabb_mins.append(aabb[:,0])  # (min_x, min_y, min_z)
    
    # すべてのgeomの中で最小Zを求める
    aabb_mins = torch.stack(aabb_mins)  # shape: (N, 3)
    lowest_z, _ = aabb_mins[:, :, 2].min(dim=0)
    return lowest_z

class BP000EnvGenesis(RLEnvGenesis):
    def __init__(self,
                 num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True,
                 device="cuda", dt=0.02, substeps=2,
                 robot_urdf_path='bp000.urdf',
                 **kwargs
                 ):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer,
                         device, dt, substeps, robot_urdf_path)
        self.l_ankle_z = torch.zeros((self.num_envs), device=self.device, dtype=torch.float32)
        self.r_ankle_z = torch.zeros((self.num_envs), device=self.device, dtype=torch.float32)
        self.max_ankle_height = torch.zeros((self.num_envs), device=self.device, dtype=torch.float32)


    def build_environment(self): ## override
        super().build_environment()

    def specific_update_buffer(self):
        self.l_ankle_z[:] = get_link_lowest_point_z(self.robot.get_link(name="L_ANKLE_R"))
        self.r_ankle_z[:] = get_link_lowest_point_z(self.robot.get_link(name="R_ANKLE_R"))
        self.max_ankle_height[:], _ = torch.max(torch.stack([self.l_ankle_z, self.r_ankle_z]), dim=0)
        return self.max_ankle_height

    def _reward_min_ankle_height(self):
       return torch.square(self.specific_update_buffer())
