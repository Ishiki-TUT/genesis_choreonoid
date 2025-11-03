import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse

# 共通設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

# 定数定義
JOINT_NAMES = ['R_HIP_Y', 'R_HIP_R', 'R_HIP_P', 'R_KNEE', 'R_ANKLE_P', 'R_ANKLE_R',
               'L_HIP_Y', 'L_HIP_R', 'L_HIP_P', 'L_KNEE', 'L_ANKLE_P', 'L_ANKLE_R']

COLORS = {
    'genesis_ang_vel': '#1f77b4',      # 青系
    'genesis_dof_pos': '#ff7f0e',      # オレンジ系
    'genesis_action': '#2ca02c',       # 緑系
    'choreonoid_ang_vel': '#d62728',   # 赤系
    'choreonoid_dof_pos': '#9467bd',   # 紫系
    'choreonoid_action': '#8c564b',    # 茶系
    # 追加: トルク
    'genesis_torque': '#17becf',       # ティール
    'choreonoid_torque': '#e377c2',    # ピンク
}

PLOT_CONFIG = {
    'figsize_large': (20, 15),
    'figsize_medium': (15, 12),
    'figsize_wide': (16, 12),
    'dpi': 300,
    'alpha_line': 0.8,
    'alpha_grid': 0.3,
    'linewidth': 2
}

# グローバル変数として保存先を定義
OUTPUT_DIR = 'obs_comparison_plots'  # デフォルト値

def load_data():
    """データ読み込み"""
    print("Loading data...")
    # genesis_df = pd.read_csv('obs_data/genesis_ishiki-walking-no-vel_ckpt2000_simple.csv')
    genesis_df = pd.read_csv('obs_data/genesis_ishiki-walking-rand_ckpt1000_scale0.csv')
    # cnoid_df = pd.read_csv('obs_data/cnoid_ishiki-walking-no-vel_ckpt2000_scale1.0.csv')
    cnoid_df = pd.read_csv('obs_data/cnoid_ishiki-walking-rand_ckpt1000_scale0.0.csv')

    print(f"Genesis data shape: {genesis_df.shape}")
    print(f"Choreonoid data shape: {cnoid_df.shape}")
    
    return genesis_df, cnoid_df

def extract_obs_components(genesis_df, cnoid_df):
    """
    観測値の各成分を抽出
    観測値の構造:
    - obs_0~2:   base_ang_vel (3)
    - obs_3~5:   projected_gravity (3)
    - obs_6~8:   commands (3)
    - obs_9~20:  dof_pos - default_dof_pos (12)
    - obs_21~32: dof_vel (12)
    - obs_33~44: actions (12)
    - torque_0~11: torques (12)  ← あれば読む
    """
    # 共通のステップ数を確認
    min_steps = min(len(genesis_df), len(cnoid_df))
    print(f"Analyzing {min_steps} steps")
    
    # Base angular velocity (obs_0~2)
    genesis_ang_vel = np.array([genesis_df[f'obs_{i}'].iloc[:min_steps] for i in range(3)]).T
    cnoid_ang_vel = np.array([cnoid_df[f'obs_{i}'].iloc[:min_steps] for i in range(3)]).T

    # Joint positions (obs_9~20)
    genesis_dof_pos = np.array([genesis_df[f'obs_{9+i}'].iloc[:min_steps] for i in range(12)]).T
    cnoid_dof_pos = np.array([cnoid_df[f'obs_{9+i}'].iloc[:min_steps] for i in range(12)]).T

    # Joint velocities (obs_21~32)
    genesis_dof_vel = np.array([genesis_df[f'obs_{21+i}'].iloc[:min_steps] for i in range(12)]).T
    cnoid_dof_vel = np.array([cnoid_df[f'obs_{21+i}'].iloc[:min_steps] for i in range(12)]).T

    # Actions (obs_33~44)
    genesis_actions = np.array([genesis_df[f'obs_{33+i}'].iloc[:min_steps] for i in range(12)]).T
    cnoid_actions = np.array([cnoid_df[f'obs_{33+i}'].iloc[:min_steps] for i in range(12)]).T

    # Torques (torque_0~11) — 存在チェックして読込
    torque_cols = [f"torque_{i}" for i in range(12)]
    has_g_torque = all((c in genesis_df.columns) for c in torque_cols)
    has_c_torque = all((c in cnoid_df.columns) for c in torque_cols)
    if has_g_torque and has_c_torque:
        # 数値化してNaNを補間→0埋め
        for col in torque_cols:
            genesis_df[col] = pd.to_numeric(genesis_df[col], errors="coerce")
            cnoid_df[col]   = pd.to_numeric(cnoid_df[col],   errors="coerce")
        genesis_df[torque_cols] = genesis_df[torque_cols].fillna(method="ffill").fillna(0.0)
        cnoid_df[torque_cols]   = cnoid_df[torque_cols].fillna(method="ffill").fillna(0.0)

        min_steps = min(len(genesis_df), len(cnoid_df))
        genesis_torque = np.array([genesis_df[f'torque_{i}'].iloc[:min_steps] for i in range(12)]).T
        cnoid_torque   = np.array([cnoid_df[f'torque_{i}'].iloc[:min_steps] for i in range(12)]).T
    else:
        genesis_torque = None
        cnoid_torque = None
        print("Info: torque_0..11 columns not found in one or both CSVs. Skipping torque plots.")

    return {
        'genesis_ang_vel': genesis_ang_vel,
        'cnoid_ang_vel': cnoid_ang_vel,
        'genesis_dof_pos': genesis_dof_pos,
        'cnoid_dof_pos': cnoid_dof_pos,
        'genesis_dof_vel': genesis_dof_vel,
        'cnoid_dof_vel': cnoid_dof_vel,
        'genesis_actions': genesis_actions,
        'cnoid_actions': cnoid_actions,
        'genesis_torque': genesis_torque,
        'cnoid_torque': cnoid_torque,
    }

def plot_base_ang_vel_comparison(data):
    """Base Angular Velocity比較 (obs_0~2)"""
    genesis_ang_vel = data['genesis_ang_vel']
    cnoid_ang_vel = data['cnoid_ang_vel']
    
    steps = np.arange(len(genesis_ang_vel))
    
    fig, axes = plt.subplots(1, 3, figsize=PLOT_CONFIG['figsize_wide'])
    ang_vel_names = ['Roll Rate', 'Pitch Rate', 'Yaw Rate']
    
    for i in range(3):
        axes[i].plot(steps, genesis_ang_vel[:, i], 
                    label='Genesis', 
                    color=COLORS['genesis_ang_vel'],
                    linewidth=PLOT_CONFIG['linewidth'],
                    alpha=PLOT_CONFIG['alpha_line'])
        
        axes[i].plot(steps, cnoid_ang_vel[:, i], 
                    label='Choreonoid', 
                    color=COLORS['choreonoid_ang_vel'],
                    linewidth=PLOT_CONFIG['linewidth'],
                    alpha=PLOT_CONFIG['alpha_line'])
        
        axes[i].set_title(f'{ang_vel_names[i]} (obs_{i})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Angular Velocity [rad/s]')
        axes[i].legend()
        axes[i].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
        
        # 統計情報を表示
        g_mean = np.mean(genesis_ang_vel[:, i])
        c_mean = np.mean(cnoid_ang_vel[:, i])
        diff_mean = np.mean(np.abs(genesis_ang_vel[:, i] - cnoid_ang_vel[:, i]))
        
        axes[i].text(0.02, 0.98, f'G_avg: {g_mean:.4f}\nC_avg: {c_mean:.4f}\nDiff: {diff_mean:.4f}', 
                    transform=axes[i].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    fontsize=8)
    
    plt.suptitle('Genesis vs Choreonoid: Base Angular Velocity Comparison\n(obs_0~2)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/base_ang_vel_comparison.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Base angular velocity comparison plot saved")

def plot_dof_pos_comparison(data):
    """Joint Position比較 (obs_9~20)"""
    genesis_dof_pos = data['genesis_dof_pos']
    cnoid_dof_pos = data['cnoid_dof_pos']
    
    steps = np.arange(len(genesis_dof_pos))
    
    fig, axes = plt.subplots(3, 4, figsize=PLOT_CONFIG['figsize_large'])
    axes = axes.flatten()
    
    for i in range(12):
        axes[i].plot(steps, genesis_dof_pos[:, i], 
                    label='Genesis', 
                    color=COLORS['genesis_dof_pos'],
                    linewidth=PLOT_CONFIG['linewidth'],
                    alpha=PLOT_CONFIG['alpha_line'])
        
        axes[i].plot(steps, cnoid_dof_pos[:, i], 
                    label='Choreonoid', 
                    color=COLORS['choreonoid_dof_pos'],
                    linewidth=PLOT_CONFIG['linewidth'],
                    alpha=PLOT_CONFIG['alpha_line'])
        
        axes[i].set_title(f'{JOINT_NAMES[i]} Position (obs_{9+i})', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Joint Position [rad]')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
        
        # 統計情報を表示
        g_mean = np.mean(genesis_dof_pos[:, i])
        c_mean = np.mean(cnoid_dof_pos[:, i])
        diff_mean = np.mean(np.abs(genesis_dof_pos[:, i] - cnoid_dof_pos[:, i]))
        
        axes[i].text(0.02, 0.98, f'G: {g_mean:.3f}\nC: {c_mean:.3f}\nΔ: {diff_mean:.4f}', 
                    transform=axes[i].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    fontsize=8)
    
    plt.suptitle('Genesis vs Choreonoid: Joint Position Comparison\n(obs_9~20: dof_pos - default_dof_pos)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dof_pos_comparison.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Joint position comparison plot saved")

def plot_dof_vel_comparison(data):
    """Joint Velocity比較 (obs_21~32)"""
    genesis_dof_vel = data['genesis_dof_vel']
    cnoid_dof_vel = data['cnoid_dof_vel']
    
    steps = np.arange(len(genesis_dof_vel))
    
    fig, axes = plt.subplots(3, 4, figsize=PLOT_CONFIG['figsize_large'])
    axes = axes.flatten()
    
    for i in range(12):
        axes[i].plot(steps, genesis_dof_vel[:, i], 
                    label='Genesis', 
                    color=COLORS['genesis_ang_vel'],  # 速度なので角速度色を使用
                    linewidth=PLOT_CONFIG['linewidth'],
                    alpha=PLOT_CONFIG['alpha_line'])
        
        axes[i].plot(steps, cnoid_dof_vel[:, i], 
                    label='Choreonoid', 
                    color=COLORS['choreonoid_ang_vel'],
                    linewidth=PLOT_CONFIG['linewidth'],
                    alpha=PLOT_CONFIG['alpha_line'])
        
        axes[i].set_title(f'{JOINT_NAMES[i]} Velocity (obs_{21+i})', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Joint Velocity [rad/s]')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
        
        # 統計情報を表示
        g_mean = np.mean(genesis_dof_vel[:, i])
        c_mean = np.mean(cnoid_dof_vel[:, i])
        diff_mean = np.mean(np.abs(genesis_dof_vel[:, i] - cnoid_dof_vel[:, i]))
        
        axes[i].text(0.02, 0.98, f'G: {g_mean:.3f}\nC: {c_mean:.3f}\nΔ: {diff_mean:.4f}', 
                    transform=axes[i].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    fontsize=8)
    
    plt.suptitle('Genesis vs Choreonoid: Joint Velocity Comparison\n(obs_21~32: dof_vel)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dof_vel_comparison.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Joint velocity comparison plot saved")

def plot_action_comparison(data):
    """Action値比較 (obs_33~44)"""
    genesis_actions = data['genesis_actions']
    cnoid_actions = data['cnoid_actions']
    
    steps = np.arange(len(genesis_actions))
    
    fig, axes = plt.subplots(3, 4, figsize=PLOT_CONFIG['figsize_large'])
    axes = axes.flatten()
    
    for i in range(12):
        axes[i].plot(steps, genesis_actions[:, i], 
                    label='Genesis', 
                    color=COLORS['genesis_action'],
                    linewidth=PLOT_CONFIG['linewidth'],
                    alpha=PLOT_CONFIG['alpha_line'])
        
        axes[i].plot(steps, cnoid_actions[:, i], 
                    label='Choreonoid', 
                    color=COLORS['choreonoid_action'],
                    linewidth=PLOT_CONFIG['linewidth'],
                    alpha=PLOT_CONFIG['alpha_line'])
        
        axes[i].set_title(f'{JOINT_NAMES[i]} Action (obs_{33+i})', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Action Value')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
        
        # 統計情報を表示
        g_mean = np.mean(genesis_actions[:, i])
        c_mean = np.mean(cnoid_actions[:, i])
        diff_mean = np.mean(np.abs(genesis_actions[:, i] - cnoid_actions[:, i]))
        
        axes[i].text(0.02, 0.98, f'G: {g_mean:.3f}\nC: {c_mean:.3f}\nΔ: {diff_mean:.4f}', 
                    transform=axes[i].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    fontsize=8)
    
    plt.suptitle('Genesis vs Choreonoid: Action Values Comparison\n(obs_33~44: actions)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/action_comparison.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Action comparison plot saved")

def plot_torque_comparison(data):
    """Torque比較 (torque_0~11)"""
    genesis_torque = data.get('genesis_torque', None)
    cnoid_torque = data.get('cnoid_torque', None)
    if genesis_torque is None or cnoid_torque is None:
        print("Skip torque comparison (no torque columns).")
        return

    steps = np.arange(len(genesis_torque))
    fig, axes = plt.subplots(3, 4, figsize=PLOT_CONFIG['figsize_large'])
    axes = axes.flatten()

    for i in range(12):
        axes[i].plot(
            steps, genesis_torque[:, i],
            label='Genesis', color=COLORS['genesis_torque'],
            linewidth=PLOT_CONFIG['linewidth'], alpha=PLOT_CONFIG['alpha_line'],
        )
        axes[i].plot(
            steps, cnoid_torque[:, i],
            label='Choreonoid', color=COLORS['choreonoid_torque'],
            linewidth=PLOT_CONFIG['linewidth'], alpha=PLOT_CONFIG['alpha_line'],
        )
        axes[i].set_title(f'{JOINT_NAMES[i]} Torque', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Torque [Nm]')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=PLOT_CONFIG['alpha_grid'])

        # 統計情報
        g_mean = np.mean(genesis_torque[:, i])
        c_mean = np.mean(cnoid_torque[:, i])
        diff_mean = np.mean(np.abs(genesis_torque[:, i] - cnoid_torque[:, i]))
        axes[i].text(
            0.02, 0.98, f'G: {g_mean:.3f}\nC: {c_mean:.3f}\nΔ: {diff_mean:.4f}',
            transform=axes[i].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            fontsize=8,
        )

    plt.suptitle('Genesis vs Choreonoid: Joint Torque Comparison\n(torque_0~11)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/torque_comparison.png',
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Torque comparison plot saved")

def plot_comprehensive_comparison(data):
    """包括的比較：各関節について3つの要素を同時表示"""
    genesis_dof_pos = data['genesis_dof_pos']
    cnoid_dof_pos = data['cnoid_dof_pos']
    genesis_dof_vel = data['genesis_dof_vel']
    cnoid_dof_vel = data['cnoid_dof_vel']
    genesis_actions = data['genesis_actions']
    cnoid_actions = data['cnoid_actions']
    
    steps = np.arange(len(genesis_dof_pos))
    
    # 最初の4関節について詳細比較
    fig, axes = plt.subplots(4, 3, figsize=PLOT_CONFIG['figsize_large'])
    
    for joint_idx in range(4):  # 最初の4関節
        joint_name = JOINT_NAMES[joint_idx]
        
        # Position (obs_9~20)
        axes[joint_idx, 0].plot(steps, genesis_dof_pos[:, joint_idx], 
                               label='Genesis', color=COLORS['genesis_dof_pos'], 
                               linewidth=2, alpha=0.8)
        axes[joint_idx, 0].plot(steps, cnoid_dof_pos[:, joint_idx], 
                               label='Choreonoid', color=COLORS['choreonoid_dof_pos'], 
                               linewidth=2, alpha=0.8)
        axes[joint_idx, 0].set_title(f'{joint_name} - Position')
        axes[joint_idx, 0].set_ylabel('Position [rad]')
        axes[joint_idx, 0].legend(fontsize=8)
        axes[joint_idx, 0].grid(True, alpha=0.3)
        
        # Velocity (obs_21~32)
        axes[joint_idx, 1].plot(steps, genesis_dof_vel[:, joint_idx], 
                               label='Genesis', color=COLORS['genesis_ang_vel'], 
                               linewidth=2, alpha=0.8)
        axes[joint_idx, 1].plot(steps, cnoid_dof_vel[:, joint_idx], 
                               label='Choreonoid', color=COLORS['choreonoid_ang_vel'], 
                               linewidth=2, alpha=0.8)
        axes[joint_idx, 1].set_title(f'{joint_name} - Velocity')
        axes[joint_idx, 1].set_ylabel('Velocity [rad/s]')
        axes[joint_idx, 1].legend(fontsize=8)
        axes[joint_idx, 1].grid(True, alpha=0.3)
        
        # Action (obs_33~44)
        axes[joint_idx, 2].plot(steps, genesis_actions[:, joint_idx], 
                               label='Genesis', color=COLORS['genesis_action'], 
                               linewidth=2, alpha=0.8)
        axes[joint_idx, 2].plot(steps, cnoid_actions[:, joint_idx], 
                               label='Choreonoid', color=COLORS['choreonoid_action'], 
                               linewidth=2, alpha=0.8)
        axes[joint_idx, 2].set_title(f'{joint_name} - Action')
        axes[joint_idx, 2].set_ylabel('Action Value')
        axes[joint_idx, 2].legend(fontsize=8)
        axes[joint_idx, 2].grid(True, alpha=0.3)
        
        # X軸ラベルは最下段のみ
        if joint_idx == 3:
            axes[joint_idx, 0].set_xlabel('Time Step')
            axes[joint_idx, 1].set_xlabel('Time Step')
            axes[joint_idx, 2].set_xlabel('Time Step')
    
    plt.suptitle('Comprehensive Comparison: Position, Velocity, Action\n(First 4 Joints)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/comprehensive_comparison.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Comprehensive comparison plot saved")

def plot_difference_analysis(data):
    """差分分析：各要素の差分を可視化"""
    genesis_dof_pos = data['genesis_dof_pos']
    cnoid_dof_pos = data['cnoid_dof_pos']
    genesis_dof_vel = data['genesis_dof_vel']
    cnoid_dof_vel = data['cnoid_dof_vel']
    genesis_actions = data['genesis_actions']
    cnoid_actions = data['cnoid_actions']
    
    # 差分計算
    pos_diff = np.abs(genesis_dof_pos - cnoid_dof_pos)
    vel_diff = np.abs(genesis_dof_vel - cnoid_dof_vel)
    action_diff = np.abs(genesis_actions - cnoid_actions)
    
    steps = np.arange(len(pos_diff))
    
    fig, axes = plt.subplots(3, 4, figsize=PLOT_CONFIG['figsize_large'])
    axes = axes.flatten()
    
    for i in range(12):
        axes[i].plot(steps, pos_diff[:, i], 
                    label='|Genesis - Choreonoid| Position', 
                    color=COLORS['genesis_dof_pos'], linewidth=2, alpha=0.8)
        
        axes[i].plot(steps, vel_diff[:, i], 
                    label='|Genesis - Choreonoid| Velocity', 
                    color=COLORS['genesis_ang_vel'], linewidth=2, alpha=0.8)
        
        axes[i].plot(steps, action_diff[:, i], 
                    label='|Genesis - Choreonoid| Action', 
                    color=COLORS['genesis_action'], linewidth=2, alpha=0.8)
        
        axes[i].set_title(f'{JOINT_NAMES[i]} - Differences', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Absolute Difference')
        axes[i].legend(fontsize=6)
        axes[i].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
        axes[i].set_yscale('log')  # 対数スケール
        
        # 平均差分を表示
        pos_mean = np.mean(pos_diff[:, i])
        vel_mean = np.mean(vel_diff[:, i])
        action_mean = np.mean(action_diff[:, i])
        
        axes[i].text(0.02, 0.02, 
                    f'Pos: {pos_mean:.4f}\nVel: {vel_mean:.4f}\nAct: {action_mean:.4f}', 
                    transform=axes[i].transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=8)
    
    plt.suptitle('Difference Analysis: |Genesis - Choreonoid| (Log Scale)\n' + 
                 'Position (obs_9~20), Velocity (obs_21~32), Action (obs_33~44)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/difference_analysis.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Difference analysis plot saved")

def print_comprehensive_statistics(data):
    """包括的な統計情報を出力"""
    genesis_ang_vel = data['genesis_ang_vel']
    cnoid_ang_vel = data['cnoid_ang_vel']
    genesis_dof_pos = data['genesis_dof_pos']
    cnoid_dof_pos = data['cnoid_dof_pos']
    genesis_dof_vel = data['genesis_dof_vel']
    cnoid_dof_vel = data['cnoid_dof_vel']
    genesis_actions = data['genesis_actions']
    cnoid_actions = data['cnoid_actions']
    genesis_torque = data.get('genesis_torque', None)
    cnoid_torque = data.get('cnoid_torque', None)
    
    print("\n" + "="*120)
    print("COMPREHENSIVE OBSERVATION COMPARISON ANALYSIS")
    print("="*120)
    
    # Base Angular Velocity統計
    print("\n--- BASE ANGULAR VELOCITY (obs_0~2) ---")
    ang_vel_names = ['Roll Rate', 'Pitch Rate', 'Yaw Rate']
    for i in range(3):
        g_mean = np.mean(genesis_ang_vel[:, i])
        c_mean = np.mean(cnoid_ang_vel[:, i])
        diff_mean = np.mean(np.abs(genesis_ang_vel[:, i] - cnoid_ang_vel[:, i]))
        print(f"{ang_vel_names[i]:<12} G:{g_mean:>8.4f} C:{c_mean:>8.4f} Diff:{diff_mean:>8.4f}")
    
    # Joint統計のヘッダー
    print(f"\n--- JOINT ANALYSIS (12 joints) ---")
    print(f"{'Joint':<12} {'G_Pos':<8} {'C_Pos':<8} {'G_Vel':<8} {'C_Vel':<8} {'G_Act':<8} {'C_Act':<8} " +
          f"{'|Pos|':<8} {'|Vel|':<8} {'|Act|':<8}")
    print("-"*120)
    
    for i in range(12):
        # 平均値計算
        g_pos_mean = np.mean(genesis_dof_pos[:, i])
        c_pos_mean = np.mean(cnoid_dof_pos[:, i])
        g_vel_mean = np.mean(genesis_dof_vel[:, i])
        c_vel_mean = np.mean(cnoid_dof_vel[:, i])
        g_act_mean = np.mean(genesis_actions[:, i])
        c_act_mean = np.mean(cnoid_actions[:, i])
        
        # 差分計算
        pos_diff = np.mean(np.abs(genesis_dof_pos[:, i] - cnoid_dof_pos[:, i]))
        vel_diff = np.mean(np.abs(genesis_dof_vel[:, i] - cnoid_dof_vel[:, i]))
        act_diff = np.mean(np.abs(genesis_actions[:, i] - cnoid_actions[:, i]))
        
        print(f"{JOINT_NAMES[i]:<12} {g_pos_mean:<8.3f} {c_pos_mean:<8.3f} " +
              f"{g_vel_mean:<8.3f} {c_vel_mean:<8.3f} {g_act_mean:<8.3f} {c_act_mean:<8.3f} " +
              f"{pos_diff:<8.4f} {vel_diff:<8.4f} {act_diff:<8.4f}")
    
    # 全体統計
    print("\n" + "-"*120)
    print("OVERALL STATISTICS")
    print("-"*120)
    
    # 全要素の平均差分
    overall_pos_diff = np.mean(np.abs(genesis_dof_pos - cnoid_dof_pos))
    overall_vel_diff = np.mean(np.abs(genesis_dof_vel - cnoid_dof_vel))
    overall_act_diff = np.mean(np.abs(genesis_actions - cnoid_actions))
    overall_ang_vel_diff = np.mean(np.abs(genesis_ang_vel - cnoid_ang_vel))
    
    print(f"Base Angular Velocity difference:  {overall_ang_vel_diff:.6f}")
    print(f"Joint Position difference:         {overall_pos_diff:.6f}")
    print(f"Joint Velocity difference:         {overall_vel_diff:.6f}")
    print(f"Action difference:                 {overall_act_diff:.6f}")
    
    if genesis_torque is not None and cnoid_torque is not None:
        overall_torque_diff = np.mean(np.abs(genesis_torque - cnoid_torque))
        print(f"Torque difference:                  {overall_torque_diff:.6f}")

    # 比率分析
    print(f"\nComponent Analysis:")
    total_diff = overall_pos_diff + overall_vel_diff + overall_act_diff
    if genesis_torque is not None and cnoid_torque is not None:
        total_diff += overall_torque_diff
        print(f"Torque contribution:   {overall_torque_diff/total_diff*100:.1f}%")
    print(f"Position contribution: {overall_pos_diff/total_diff*100:.1f}%")
    print(f"Velocity contribution: {overall_vel_diff/total_diff*100:.1f}%")
    print(f"Action contribution:   {overall_act_diff/total_diff*100:.1f}%")

def main():
    global OUTPUT_DIR  # グローバル変数を使用
    
    parser = argparse.ArgumentParser(description="Genesis vs Choreonoid observation comparison")
    parser.add_argument("-o", "--output", type=str, default="obs_comparison_plots",
                        help="Output directory for plots (default: obs_comparison_plots)")
    parser.add_argument("-gene", "--genesis-file", type=str, 
                        default="obs_data/genesis_ishiki-walking-no-vel_ckpt2000_simple.csv",
                        help="Genesis data file path")
    parser.add_argument("-cnoid", "--choreonoid-file", type=str,
                        default="obs_data/cnoid_ishiki-walking-no-vel_ckpt2000_simple.csv", 
                        help="Choreonoid data file path")
    args = parser.parse_args()
    
    # 出力ディレクトリを設定
    OUTPUT_DIR = args.output
    
    # 出力ディレクトリを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    
    """メイン実行関数"""
    print("=" * 80)
    print("GENESIS vs CHOREONOID: OBSERVATION COMPARISON ANALYSIS")
    print("=" * 80)
    print("Analysis includes:")
    print("1. Base Angular Velocity (obs_0~2)")
    print("2. Joint Positions (obs_9~20)")
    print("3. Joint Velocities (obs_21~32)")
    print("4. Action Values (obs_33~44)")
    print("-" * 80)
    
    # データ読み込みと抽出
    genesis_df, cnoid_df = load_data()
    data = extract_obs_components(genesis_df, cnoid_df)

    # 生成するプロット
    plot_functions = [
        plot_base_ang_vel_comparison,
        plot_dof_pos_comparison,
        plot_dof_vel_comparison,
        plot_action_comparison,
        plot_comprehensive_comparison,
        plot_difference_analysis,
    ]
    # トルクがあれば追加
    if data.get('genesis_torque') is not None and data.get('cnoid_torque') is not None:
        plot_functions.insert(3, plot_torque_comparison)  # アクションの前後どちらでもOK

    for plot_func in plot_functions:
        plot_func(data)

    # 統計サマリーを出力
    print_comprehensive_statistics(data)
    
    print("\n" + "=" * 80)
    print(f"All plots saved in '{OUTPUT_DIR}/' directory:")
    print("1. base_ang_vel_comparison.png")
    print("2. dof_pos_comparison.png")
    print("3. dof_vel_comparison.png")
    if data.get('genesis_torque') is not None and data.get('cnoid_torque') is not None:
        print("4. torque_comparison.png")
        print("5. action_comparison.png")
        print("6. comprehensive_comparison.png")
        print("7. difference_analysis.png")
    else:
        print("4. action_comparison.png")
        print("5. comprehensive_comparison.png")
        print("6. difference_analysis.png")
    print("=" * 80)


if __name__ == "__main__":  main()

"""
# 使用例:
# データ収集付き評価(100ステップ)
python3 ishiki_Errorverifi.py -e ishiki-walking-rand -o obs_comparison_plots_rand 
--genesis-file obs_data/genesis_ishiki-walking-rand_ckpt1000_simple.csv --choreonoid-file obs_data/cnoid_ishiki-walking-rand_ckpt1000_scale1.0.csv

"""
