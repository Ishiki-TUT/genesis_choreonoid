"""
Stepping Controller - C++ の stepping_controller.cpp を Python に変換
リアルタイムの足の軌跡制御と DCM ベースのステップ調整
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.spatial.transform import Rotation as R
from footstep_planner import Step, Footstep, Param, Ground, rotate_vector

pi = 3.14159265358979
eps = 1.0e-10


@dataclass
class Timer:
    """タイマー情報"""
    time: float = 0.0


@dataclass
class Centroid:
    """重心情報"""
    dcm_ref: np.ndarray = None       # 参考 DCM
    dcm_target: np.ndarray = None    # 目標 DCM
    zmp_ref: np.ndarray = None       # 参考 ZMP
    zmp_target: np.ndarray = None    # 目標 ZMP
    
    def __post_init__(self):
        if self.dcm_ref is None:
            self.dcm_ref = np.array([0.0, 0.0, 0.8])
        if self.dcm_target is None:
            self.dcm_target = np.array([0.0, 0.0, 0.8])
        if self.zmp_ref is None:
            self.zmp_ref = np.array([0.0, 0.0, 0.0])
        if self.zmp_target is None:
            self.zmp_target = np.array([0.0, 0.0, 0.0])


@dataclass
class Base:
    """ベース（ボディ）情報"""
    angle: np.ndarray = None         # 現在の角度 [roll, pitch, yaw]
    angle_ref: np.ndarray = None     # 参考角度
    ori: R = None                    # 現在の向き
    ori_ref: R = None                # 参考向き
    
    def __post_init__(self):
        if self.angle is None:
            self.angle = np.array([0.0, 0.0, 0.0])
        if self.angle_ref is None:
            self.angle_ref = np.array([0.0, 0.0, 0.0])
        if self.ori is None:
            self.ori = R.from_euler('xyz', [0, 0, 0])
        if self.ori_ref is None:
            self.ori_ref = R.from_euler('xyz', [0, 0, 0])


@dataclass
class Foot:
    """足の情報"""
    pos_ref: np.ndarray = None       # 参考位置
    angle_ref: np.ndarray = None     # 参考角度 [roll, pitch, yaw]
    ori_ref: R = None                # 参考向き
    contact_ref: bool = True         # 接触フラグ
    
    def __post_init__(self):
        if self.pos_ref is None:
            self.pos_ref = np.array([0.0, 0.0, 0.0])
        if self.angle_ref is None:
            self.angle_ref = np.array([0.0, 0.0, 0.0])
        if self.ori_ref is None:
            self.ori_ref = R.from_euler('xyz', [0, 0, 0])


class SteppingController:
    """リアルタイム足の軌跡制御"""
    
    def __init__(self):
        self.swing_height = 0.05              # スウィング足の高さ
        self.swing_tilt = 0.0                 # スウィング足の傾き
        self.dsp_duration = 0.1               # ダブルサポート期間
        self.descend_duration = 0.0           # 降下期間
        self.descend_depth = 0.0              # 降下深さ
        self.timing_adaptation_weight = 1.0   # タイミング適応の重み
        
        self.buffer_ready = False
        self.time_to_landing = 0.0
    
    def update(self, timer: Timer, param: Param, footstep: Footstep, footstep_buffer: Footstep, centroid: Centroid, base: Base, foot: List[Foot]):
        T = param.T
        offset = np.array([0.0, 0.0, param.com_height])
        
        # 最初にステップ数の確認と、正しい参照（st0, st1, stb0, stb1）の固定
        if len(footstep.steps) < 2 or len(footstep_buffer.steps) < 2:
            return
            
        st0 = footstep.steps[0]
        st1 = footstep.steps[1]
        stb0 = footstep_buffer.steps[0]
        stb1 = footstep_buffer.steps[1]
        
        sup = st0.side
        swg = 1 - st0.side
        
        if self.buffer_ready:
            t_ref = timer.time - stb0.tbegin
            alpha_ref = np.exp(t_ref / T)
            
            xi0 = stb0.dcm[:2] - stb0.zmp[:2]
            xi = centroid.dcm_ref[:2] - stb0.zmp[:2]
            
            w = self.timing_adaptation_weight
            alpha = (w * w * alpha_ref + np.linalg.norm(xi0) * np.linalg.norm(xi)) / (w * w + np.dot(xi0, xi0))
            t_dcm = T * np.log(alpha)
            
            self.time_to_landing = stb0.duration - t_dcm
            
            if self.time_to_landing <= 0.0:
                if len(footstep.steps) > 1:
                    footstep.steps.pop(0)
                    if len(footstep.steps) == 1:
                        print("end of footstep reached")
                        return
                
                footstep_buffer.steps[1].dcm = footstep_buffer.steps[0].dcm.copy()
                footstep_buffer.steps.pop(0)
                footstep_buffer.steps.append(Step())
                
                self.buffer_ready = False
                
                # バッファがポップされたため、stb0, stb1 を再取得
                stb0 = footstep_buffer.steps[0]
                stb1 = footstep_buffer.steps[1]
            else:
                centroid.dcm_target = (stb0.zmp + offset) + alpha_ref * (stb0.dcm - (stb0.zmp + offset))
        
        if not self.buffer_ready:
            stb0.side = st0.side
            stb1.side = st1.side
            stb0.stepping = st0.stepping
            stb0.duration = st0.duration
            
            stb0.foot_pos[sup] = foot[sup].pos_ref.copy()
            stb0.foot_angle[sup] = np.array([0.0, 0.0, foot[sup].angle_ref[2]])
            stb0.foot_ori[sup] = R.from_euler('xyz', stb0.foot_angle[sup])
            
            stb0.foot_pos[swg] = foot[swg].pos_ref.copy()
            stb0.foot_angle[swg] = np.array([0.0, 0.0, foot[swg].angle_ref[2]])
            stb0.foot_ori[swg] = R.from_euler('xyz', stb0.foot_angle[swg])
            
            stb0.dcm = centroid.dcm_ref.copy()
            
            # 【バグ②の修正】基準を stb0 ではなく 計画値 st0 に変更
            ori_rel_inv = st0.foot_ori[sup].inv()
            ori_rel = ori_rel_inv * st1.foot_ori[swg]
            pos_rel = ori_rel_inv.apply(st1.foot_pos[swg] - st0.foot_pos[sup])
            dcm_rel = ori_rel_inv.apply(st1.dcm - st0.foot_pos[sup])
            
            stb1.foot_pos[sup] = stb0.foot_pos[sup].copy()
            stb1.foot_ori[sup] = stb0.foot_ori[sup]
            stb1.foot_angle[sup] = stb0.foot_angle[sup].copy()
            
            stb1.foot_pos[swg] = stb0.foot_pos[sup] + stb0.foot_ori[sup].apply(pos_rel)
            stb1.foot_ori[swg] = stb0.foot_ori[sup] * ori_rel
            stb1.foot_angle[swg] = stb1.foot_ori[swg].as_euler('xyz')
            
            stb1.dcm = stb0.foot_pos[sup] + stb0.foot_ori[sup].apply(dcm_rel)
            
            alpha = np.exp(stb0.duration / T)
            if abs(alpha - 1.0) > eps:
                stb0.zmp = (1.0 / (alpha - 1.0)) * (alpha * stb0.dcm - stb1.dcm) - offset
            else:
                stb0.zmp = stb0.dcm.copy()
            
            centroid.zmp_target = stb0.zmp.copy()
            stb0.tbegin = timer.time
            self.time_to_landing = stb0.duration
            self.buffer_ready = True
            
        # 着地時の DCM を予測
        land_dcm = (stb0.zmp + offset) + np.exp(self.time_to_landing / T) * \
                   (centroid.dcm_ref - (stb0.zmp + offset))
        
        # 着地調整（DCM ベース） - ここで st1 が本来の footstep.steps[1] を正しく参照するようになります
        stb1.foot_pos[swg][0] = land_dcm[0] - (st1.dcm[0] - st1.foot_pos[swg][0])
        stb1.foot_pos[swg][1] = land_dcm[1] - (st1.dcm[1] - st1.foot_pos[swg][1])
        
        # ベース向きは足の向きの中点
        angle_diff = foot[1].angle_ref[2] - foot[0].angle_ref[2]
        while angle_diff > pi: angle_diff -= 2.0 * pi
        while angle_diff < -pi: angle_diff += 2.0 * pi
        base.angle_ref[2] = foot[0].angle_ref[2] + angle_diff / 2.0
        base.ori_ref = R.from_euler('xyz', base.angle_ref)
        
        # サポート足の位置を設定
        foot[sup].pos_ref = stb0.foot_pos[sup].copy()
        foot[sup].angle_ref = stb0.foot_angle[sup].copy()
        foot[sup].ori_ref = R.from_euler('xyz', foot[sup].angle_ref)
        foot[sup].contact_ref = True
        
        # スウィング足の位置を設定
        if not stb0.stepping or self.time_to_landing > (stb0.duration - self.dsp_duration):
            foot[swg].pos_ref = stb0.foot_pos[swg].copy()
            foot[swg].angle_ref = stb0.foot_angle[swg].copy()
            foot[swg].ori_ref = stb0.foot_ori[swg]
            foot[swg].contact_ref = True
        else:
            ts = (stb0.duration - self.dsp_duration) - self.time_to_landing
            tauv = stb0.duration - self.dsp_duration
            tauh = tauv - self.descend_duration
            
            sv = ts / tauv if tauv > 0 else 0
            sh = ts / tauh if tauh > 0 else 0
            thetav = 2.0 * pi * sv
            thetah = 2.0 * pi * sh
            
            ch = (thetah - np.sin(thetah)) / (2.0 * pi) if sh < 1.0 else 1.0
            cv = (1.0 - np.cos(thetav)) / 2.0
            cv2 = (1.0 - np.cos(thetav / 2.0)) / 2.0
            cw = np.sin(thetah)
            
            turn = stb1.foot_angle[swg] - stb0.foot_angle[swg]
            while turn[2] > pi: turn[2] -= 2.0 * pi
            while turn[2] < -pi: turn[2] += 2.0 * pi
            
            tilt = stb0.foot_ori[swg].apply(np.array([0.0, self.swing_tilt, 0.0]))
            
            foot[swg].pos_ref = (1.0 - ch) * stb0.foot_pos[swg] + ch * stb1.foot_pos[swg]
            foot[swg].pos_ref[2] += (cv * (self.swing_height + 0.5 * self.descend_depth) - cv2 * self.descend_depth)
            foot[swg].angle_ref = stb0.foot_angle[swg] + ch * turn + cw * tilt
            foot[swg].ori_ref = R.from_euler('xyz', foot[swg].angle_ref)
            foot[swg].contact_ref = False
            
            # 【バグ③の修正】SciPyの乗算規則（左が先、右が後）に合わせて C++ (Q_act.inv() が先、Q_ref が後) を表現
            qrel = R.from_euler('xyz', base.angle_ref) * R.from_euler('xyz', [base.angle[0], base.angle[1], base.angle_ref[2]]).inv()
            pivot = centroid.zmp_ref
            
            foot[swg].pos_ref = qrel.apply(foot[swg].pos_ref - pivot) + pivot
            foot[swg].ori_ref = qrel * foot[swg].ori_ref
            foot[swg].angle_ref = foot[swg].ori_ref.as_euler('xyz')