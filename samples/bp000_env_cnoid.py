import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../irsl_rl')

from rl_env_cnoid import RLEnvChoreonoid

class BP000Env(RLEnvChoreonoid):
    def __init__(self,
                 num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True,
                 device="cuda", dt=0.02, substeps=20,
                 robot_urdf_path='bp000.urdf'):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer,
                         device, dt, substeps, robot_urdf_path)

#    def build_environment(self): ## override
#        # add plain
#        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
