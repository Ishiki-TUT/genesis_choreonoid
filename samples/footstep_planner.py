"""
Footstep Planner - C++ の footstep_planner.cpp を Python に変換
歩行ステップの計画と生成
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.spatial.transform import Rotation as R

eps = 1.0e-10


@dataclass
class Step:
    """1ステップの情報"""
    # 足の位置と姿勢（左足[0]、右足[1]）
    foot_pos: np.ndarray = None      # (2, 3) - 足の位置 [left, right]
    foot_angle: np.ndarray = None    # (2, 3) - roll, pitch, yaw [left, right]
    foot_ori: list = None            # (2,) - 回転行列 [left, right]
    
    # 足の側（0: 左, 1: 右）
    side: int = 0
    
    # ステップパラメータ
    stride: float = 0.2              # 前進距離
    sway: float = 0.0                # 横揺れ
    turn: float = 0.0                # 回転角度
    spacing: float = 0.2             # 足の間隔
    climb: float = 0.0               # 階段登り
    
    # タイミング情報
    duration: float = 0.8            # ステップの継続時間
    tbegin: float = 0.0              # ステップ開始時刻
    
    # DCM と ZMP
    dcm: np.ndarray = None           # Divergent Component of Motion
    zmp: np.ndarray = None           # Zero Moment Point
    
    # その他
    stepping: bool = True            # ステップするか（サポート交換）
    
    def __post_init__(self):
        if self.foot_pos is None:
            self.foot_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # (2, 3)
        if self.foot_angle is None:
            self.foot_angle = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # (2, 3)
        if self.foot_ori is None:
            self.foot_ori = [R.from_euler('xyz', [0, 0, 0]), R.from_euler('xyz', [0, 0, 0])]
        if self.dcm is None:
            self.dcm = np.array([0.0, 0.0, 0.0])
        if self.zmp is None:
            self.zmp = np.array([0.0, 0.0, 0.0])


@dataclass
class Footstep:
    """複数ステップの歩行計画"""
    steps: List[Step] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []


@dataclass
class Param:
    """歩行パラメータ"""
    com_height: float = 0.8          # CoM の高さ
    T: float = 1.0                   # 時定数


@dataclass
class Ground:
    """地面情報"""
    ori: R = None                    # 地面の姿勢
    
    def __post_init__(self):
        if self.ori is None:
            self.ori = R.from_euler('xyz', [0, 0, 0])


def quat_to_rpy(quat):
    """クォータニオンから RPY を取得"""
    r = R.from_quat(quat)
    rpy = r.as_euler('xyz')
    return rpy


def rpy_to_quat(rpy):
    """RPY からクォータニオンを生成"""
    r = R.from_euler('xyz', rpy)
    return r.as_quat()


def rotate_vector(ori, vec):
    """ベクトルを回転行列で回転"""
    return ori.apply(vec)


class FootstepPlanner:
    """歩行ステップの計画"""
    
    def __init__(self):
        pass
    
    def plan(self, param: Param, footstep: Footstep):
        """
        足の配置とサポート足フラグを決定
        
        Args:
            param: 歩行パラメータ
            footstep: 歩行計画（step[0]の足の配置とDCMは外部で指定される）
        """
        nstep = len(footstep.steps)
        
        for i in range(nstep - 1):
            st0 = footstep.steps[i]
            st1 = footstep.steps[i + 1]
            
            sup = st0.side          # サポート足
            swg = 1 - st0.side      # スウィング足
            
            dtheta = st0.turn
            l = st0.stride
            d = st0.sway
            w = (1.0 if sup == 0 else -1.0) * st0.spacing
            dz = st0.climb
            
            # スウィング足の相対位置を計算
            if abs(dtheta) < eps:
                dprel = np.array([l, w + d, dz])
            else:
                r = l / dtheta
                dprel = np.array([
                    (r - w / 2.0 - d) * np.sin(dtheta),
                    (r + w / 2.0) - (r - w / 2.0 - d) * np.cos(dtheta),
                    dz
                ])
            
            # サポート足と スウィング足を交換
            st1.side = 1 - st0.side
            
            # サポート足の位置は変わらない
            st1.foot_pos[sup] = st0.foot_pos[sup].copy()
            st1.foot_angle[sup] = st0.foot_angle[sup].copy()
            st1.foot_ori[sup] = st0.foot_ori[sup]
            
            # スウィング足の位置が変わる
            st1.foot_pos[swg] = st0.foot_pos[sup] + rotate_vector(st0.foot_ori[sup], dprel)
            st1.foot_angle[swg] = st0.foot_angle[sup] + np.array([0.0, 0.0, dtheta])
            st1.foot_ori[swg] = R.from_euler('xyz', st1.foot_angle[swg])
    
    def align_to_ground(self, ground: Ground, footstep: Footstep):
        """
        地面に足を合わせる
        
        Args:
            ground: 地面情報
            footstep: 歩行計画
        """
        # 初期サポート足の中心を基準に回転
        pivot = footstep.steps[0].foot_pos[footstep.steps[0].side]
        
        # 地面の法線ベクトル
        normal = rotate_vector(ground.ori, np.array([0.0, 0.0, 1.0]))
        
        for k in range(len(footstep.steps)):
            st = footstep.steps[k]
            
            for i in range(2):
                # Z座標を修正
                dp = st.foot_pos[i] - pivot
                if abs(normal[2]) > eps:
                    dp[2] = -(normal[0] * dp[0] + normal[1] * dp[1]) / normal[2]
                
                st.foot_pos[i][2] = pivot[2] + dp[2]
                
                # 地面法線を足の yaw ローカル座標に変換
                yaw = st.foot_angle[i][2]
                rot_z_neg = R.from_euler('z', -yaw)
                nl = rotate_vector(rot_z_neg, normal)
                
                st.foot_angle[i][0] = np.arcsin(-nl[1])
                st.foot_angle[i][1] = np.arctan2(nl[0], nl[2])
                
                # クォータニオンに変換
                st.foot_ori[i] = R.from_euler('xyz', st.foot_angle[i])
    
    def generate_dcm(self, param: Param, footstep: Footstep):
        """
        参考 DCM と ZMP を生成
        
        Args:
            param: 歩行パラメータ
            footstep: 歩行計画
        """
        nstep = len(footstep.steps)
        offset = np.array([0.0, 0.0, param.com_height])
        
        # 最後のステップの状態を設定
        i = nstep - 1
        # ZMP は足の中点
        footstep.steps[i].zmp = (footstep.steps[i].foot_pos[0] + footstep.steps[i].foot_pos[1]) / 2.0
        
        # DCM は ZMP から com_height 上
        footstep.steps[i].dcm = (footstep.steps[i].foot_pos[0] + footstep.steps[i].foot_pos[1]) / 2.0 + offset
        
        i -= 1
        
        # N-1 から 0 ステップの状態を計算
        while i >= 0:
            st0 = footstep.steps[i]
            st1 = footstep.steps[i + 1]
            
            sup = st0.side
            swg = 1 - st0.side
            
            a = np.exp(-st0.duration / param.T)
            
            # 初期ステップ: DCM は外部で指定済み、ZMP を決定
            if i == 0:
                st0.zmp = (st0.dcm - a * st1.dcm) / (1.0 - a) - offset
            else:
                # その他のステップ
                eps_local = 1.0e-3
                
                # スウィング足の位置が変わらない場合はダブルサポート
                if (np.linalg.norm(st0.foot_pos[swg] - st1.foot_pos[swg]) < eps_local and
                    np.linalg.norm(st0.foot_angle[swg] - st1.foot_angle[swg]) < eps_local):
                    st0.zmp = (st0.foot_pos[sup] + st0.foot_pos[swg]) / 2.0
                else:
                    # 그렇지 않으면 ZMP をサポート足に設定
                    st0.zmp = st0.foot_pos[sup].copy()
                
                # DCM を ZMP から決定
                st0.dcm = (1.0 - a) * (st0.zmp + offset) + a * st1.dcm
            
            # ステップフラグを設定
            eps_local = 1.0e-3
            if (np.linalg.norm(st0.foot_pos[swg] - st1.foot_pos[swg]) < eps_local and
                np.linalg.norm(st0.foot_angle[swg] - st1.foot_angle[swg]) < eps_local):
                st0.stepping = False
            else:
                st0.stepping = True
            
            i -= 1
    
    def generate_5step_walking_plan(stride=0.05, sway=0.0, turn=0.0, 
                                     spacing=0.2, com_height=0.65, T=1.0, 
                                     duration_per_step=0.8):
        """
        5ステップの歩行計画を生成
        
        Args:
            stride: 前進距離 [m]
            sway: 左右のスウェイ幅 [m]
            turn: 旋回角度 [rad]
            spacing: 足間隔 [m]
            com_height: CoM高さ [m]
            T: 時間定数 [s]
            duration_per_step: 1ステップの時間 [s]
        
        Returns:
            footstep: Footstepオブジェクト（複数のStepを含む）
            param: Paramオブジェクト
        """
        
        param = Param(com_height=com_height, T=T)
        steps = []
        
        # 初期位置
        current_x = 0.0
        current_y = 0.0
        current_time = 0.0
        
        print("\n" + "="*70)
        print("Debug: Generating 5-step walking plan")
        print(f"  stride={stride}, spacing={spacing}, duration_per_step={duration_per_step}")
        print("="*70)
        
        # ★重要: ステップを逐次的に生成（5ステップ）
        for step_idx in range(5):
            step = Step()
            
            # 【重要】各ステップの位置を計算
            if step_idx == 0:
                # ステップ0（初期左足）
                current_x = 0.0
                current_y = spacing / 2  # 左足は+Y
                step.stepping = False
                
            elif step_idx == 1:
                # ステップ1（右足）- 前進
                current_x += stride
                current_y = -spacing / 2  # 右足は-Y
                step.stepping = True
                
            elif step_idx == 2:
                # ステップ2（左足）- 前進
                current_x += stride
                current_y = spacing / 2  # 左足は+Y
                step.stepping = True
                
            elif step_idx == 3:
                # ステップ3（右足）- 前進
                current_x += stride
                current_y = -spacing / 2  # 右足は-Y
                step.stepping = True
                
            elif step_idx == 4:
                # ステップ4（左足）- 前進
                current_x += stride
                current_y = spacing / 2  # 左足は+Y
                step.stepping = True
            
            # ZMP位置を設定
            step.zmp = np.array([current_x, current_y, 0.0], dtype=np.float32)
            
            # DCM位置を初期値として設定
            step.dcm = step.zmp.copy()
            
            # 時間を設定
            step.time = current_time
            current_time += duration_per_step
            
            steps.append(step)
            
            # デバッグ出力
            foot_name = "LEFT" if step_idx % 2 == 0 else "RIGHT"
            print(f"Step {step_idx} ({foot_name}): ZMP=[{step.zmp[0]:.4f}, {step.zmp[1]:.4f}], time={step.time:.2f}s, stepping={step.stepping}")
        
        print(f"Total steps generated: {len(steps)}")
        print("="*70 + "\n")
        
        footstep = Footstep(steps=steps)
        
        return footstep, param


# ★重要: クラス定義の外側に配置（クラスの __init__ や __post_init__ ではなく）
