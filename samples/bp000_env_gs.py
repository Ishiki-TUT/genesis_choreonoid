import sys
import os

from rl_env_gs import RLEnvGenesis
# from rl_env_gs_tr import RLEnvGenesis
import genesis as gs
import torch
import numpy as np

from genesis.utils.geom import transform_quat_by_quat, quat_to_xyz, transform_by_quat, inv_quat


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
    
    # def _reward_min_ankle_height(self):
    #    return torch.exp(self.specific_update_buffer())  squareかexpか選択

##追加リワード
    def _reward_ankle_regularization(self):
        """
        足首のPitch（前後）とRoll（左右）の傾きを罰する
        """
        # 左右の足リンク名（URDFに合わせて確認してください）
        link_names = ["L_ANKLE_R", "R_ANKLE_R"]
        
        total_penalty = 0.0
        
        for name in link_names:
            link = self.robot.get_link(name)
            quat = link.get_quat() # (num_envs, 4)
            
            # クォータニオンをオイラー角(Roll, Pitch, Yaw)に変換
            # rpy=True で (roll, pitch, yaw) の順で返ってくると仮定
            rpy = quat_to_xyz(quat, rpy=True)

            roll = rpy[:, 0]
            pitch = rpy[:, 1]
            
            # --- 罰則の計算 ---
            # Roll (左右): 常に水平であってほしいので、厳しく罰する
            roll_penalty = torch.square(roll)
            
            # Pitch (前後): 歩行サイクル上、ある程度は動くので係数を弱めるか、
            # あるいは「大きすぎる傾き」だけを罰する
            pitch_penalty = torch.square(pitch)
            
            # ここで重み付けを変えて合算
            # 例: Rollは全力で止める(1.0)、Pitchは少し許容する(0.1)
            total_penalty += torch.sum(roll_penalty + 0.1 * pitch_penalty)
            
        return total_penalty

    def _reward_feet_stride(self):
            """
            【歩幅のリワード】
            左右の足が「進行方向（X軸）」にどれだけ離れているかを評価します。
            これが大きい＝Hip Pitchを使って足を前に出している、ということになります。
            """
            foot_names = self.env_cfg.get("feet_link_names", ["L_ANKLE_R", "R_ANKLE_R"])
            
            # 足の位置を取得
            feet_pos = []
            for name in foot_names:
                link = self.robot.get_link(name)
                # ローカル座標ではなくワールド座標でOK（進行方向がX軸と仮定）
                # もしロボットが回転する場合は、ベース座標系への変換が必要ですが、
                # 学習初期はワールドX軸の距離を見るだけで十分機能します。
                feet_pos.append(link.get_pos())
                
            if len(feet_pos) < 2:
                return 0.0
                
            # 左足と右足の位置
            pos1 = feet_pos[0]
            pos2 = feet_pos[1]
            
            # 進行方向(X軸)の距離の絶対値
            # 前後に開けば開くほど報酬が増える
            stride = torch.abs(pos1[:, 0] - pos2[:, 0])
            
            # ただし、ある程度以上（例: 0.5m）開いたらそれ以上は求めない（股裂き防止）
            return torch.clamp(stride, max=0.5)

    def _reward_hip_pitch_motion(self):
        """
        【Hip Pitchの可動域リワード】
        特定の関節名を直接指定して、その関節が大きく動くことを推奨する。
        """
        # 初回のみインデックスを特定してキャッシュ
        if not hasattr(self, "_hip_pitch_indices"):
            all_joints = self.env_cfg["joint_names"]
            
            # ★ここに計算対象の関節名を直接記述します
            target_joints = ["R_HIP_P", "L_HIP_P"]
            
            indices = []
            for name in target_joints:
                if name in all_joints:
                    idx = all_joints.index(name)
                    indices.append(idx)
                else:
                    # 万が一名前が間違っていた場合の警告
                    print(f"[Warning] Joint '{name}' not found in robot joint list.")
            
            # GPU上のTensorとして保存
            self._hip_pitch_indices = torch.tensor(indices, device=self.device, dtype=torch.long)
            
        if len(self._hip_pitch_indices) == 0:
            return 0.0
            
        # 指定関節の角度の二乗平均（動きを推奨）
        target_angles = self.dof_pos[:, self._hip_pitch_indices]
        return torch.mean(torch.square(target_angles), dim=1)
    
    def _reward_feet_alternating_pos(self):
        """交互歩行リワード（改善版）"""
        foot_names = self.env_cfg.get("feet_link_names", ["L_ANKLE_R", "R_ANKLE_R"])
        inv_base_quat = inv_quat(self.base_quat)
        
        feet_local_x = []
        for name in foot_names:
            link = self.robot.get_link(name)
            foot_world_pos = link.get_pos()
            rel_pos = foot_world_pos - self.base_pos
            local_pos = transform_by_quat(rel_pos, inv_base_quat)
            feet_local_x.append(local_pos[:, 0])
            
        if len(feet_local_x) < 2:
            return torch.zeros(self.num_envs, device=self.device)
            
        pos_product = feet_local_x[0] * feet_local_x[1]
        
        # 報酬を上限付きにする（過度な開脚を防ぐ）
        reward = torch.clamp(-pos_product, max=0.1)
        
        # 移動中のみ有効（静止時は無効化）
        moving = torch.norm(self.commands[:, :2], dim=1) > 0.1
        reward = reward * moving.float()
        
        return reward
