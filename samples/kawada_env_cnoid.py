import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../irsl_rl')
#sys.path.append('/home/irsl/Documents/genesis_choreonoid/irsl_rl')

from rl_env_cnoid import RLEnvChoreonoid

class KawadaBaseEnvChoreonoid(RLEnvChoreonoid):
    def __init__(self,
                 num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True,
                 device="cuda", dt=0.02, substeps=2,
                 robot_urdf_path='../userdir/humanoid_research_k/robots/kawada_base_500.urdf',
                 **kwargs
                 ):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer,
                         device, dt, substeps, robot_urdf_path)

    def build_environment(self): ## override
        super().build_environment()
        # self.scene.add_entity(gs.morphs.URDF(file="env.urdf", fixed=True))