import torch
import numpy as np

from rl_env_base import RLEnvBase

exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())
from irsl_choreonoid.simulation_utils import SimulationEnvironment

class RLEnvChoreonoid(RLEnvBase):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True,
                 device="cuda", dt=0.02, substeps=2,
                 robot_urdf_path=None, **kwargs):

        self.cnoid_model_class = None
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, device, dt, substeps, robot_urdf_path)

    def build_environment(self):
        ib.loadRobotItem(cutil.getShareDirectory() + '/model/misc/floor.body')
        # ib.loadRobotItem(cutil.getShareDirectory() + '/model/misc/floor3.body') # 反発係数0指定の床

    def scene_build(self, substeps, robot_urdf_path, show_viewer):
        #self.srobot = setupSim()
        if self.cnoid_model_class is not None:
            self.srobot = self.cnoid_model_class.robot_class(world=True, name='CnoidRobot') ##
        else:
            self.srobot = RobotModel.loadModelItem(robot_urdf_path, world=True, name='CnoidRobot') ##

        ### build environments geometries
        self.build_environment()

        #self.sim = simEnv()
        self.sim = SimulationEnvironment('CnoidRobot')
        self.sim.stop()

        self.srobot.setAngleMap( self.env_cfg["default_joint_angles"] )

        cds = coordinates(self.base_init_pos.cpu().numpy())
        cds.quaternion_wxyz = self.base_init_quat.cpu().numpy()
        self.srobot.rootCoords( cds )
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32) ## to be fixed
        self.srobot.item.storeInitialState()

    def startSim(self):
        self.sim.start(dt = self.dt/self.substeps, P = self.env_cfg["kp"], D = self.env_cfg["kd"],
                       simple = True,
                       effortRange = self.env_cfg['effortRange'] if 'effortRange' in self.env_cfg else True,
                       rotorInertia = self.env_cfg['rotorInertia'] if 'rotorInertia' in self.env_cfg else None,
                       )
        self.init_cds = self.sim.sbody.rootLink.getCoords()

    def env_step(self): ## override
        #self.sim.sequencer.setNoInterpolation([ self.target_dof_pos[0].cpu().numpy() ], 1)
        self.sim.sequencer.setNoInterpolation([ self.target_dof_pos[0].cpu().numpy() ], self.substeps)
        self.sim.runCount(self.substeps)

    def update_buffers(self): ## override
        self.episode_length_buf += 1

        sbody = self.sim.sbody
        root_link = sbody.rootLink
        base_coords = root_link.getCoords()

        # self.base_pos[:] = self.robot.get_pos()
        self.base_pos[0] = torch.tensor(base_coords.pos, device=self.device)
        # self.base_quat[:] = self.robot.get_quat()
        self.base_quat[0] = torch.tensor(base_coords.quaternion_wxyz, device=self.device)
        # self.base_euler = quat_to_xyz(
        #     transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
        #     rpy=True,
        #     degrees=True,
        # )
        self.base_euler[0] = torch.rad2deg(torch.tensor(base_coords.copy().transform(self.init_cds.copy().inverse()).getRPY()))

        # self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        ## self.base_lin_vel = ...
        # self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        # self.base_ang_vel[:] = torch.tensor([root_link.w]).to(torch.float).to(self.device)
        self.base_ang_vel[:]  = torch.tensor(base_coords.inverse_rotate_vector(np.array(root_link.w)), device=self.device)
        # print(self.base_ang_vel)
        # self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        # base_quat_gs = np.array([[base_quat[3] , base_quat[0], base_quat[1], base_quat[2]]])
        # inv_base_quat = torch.tensor(inv_quat(base_quat_gs)).to(gs.tc_float).to(device)
        self.projected_gravity[0] = torch.tensor(base_coords.inverse_rotate_vector(np.array([0,0,-1])), device=self.device)
        # self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_pos = torch.tensor([sbody.angleVector()]).to(torch.float32).to(self.device)
        # self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.dof_vel = torch.tensor([sbody.getVelocities()]).to(torch.float32).to(self.device)
        # self.dof_force[:] = self.robot.get_dofs_force(self.motors_dof_idx)
        # self.dof_force = ...
        # self.l_ankle_z[:] = self.robot.get_link(name="L_ANKLE_R").get_pos()[:,2]
        # self.r_ankle_z[:] = self.robot.get_link(name="R_ANKLE_R").get_pos()[:,2]
        # self.r_ankle_z = ...
        # self.min_ankle_height[:], _ = torch.min(torch.stack([self.l_ankle_z,self.r_ankle_z]), dim=0)

    def reset_env_idx(self, envs_idx): ## override
        ### start-simulation
        self.startSim()
