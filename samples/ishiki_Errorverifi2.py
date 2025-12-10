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
    # Genesis (青系)
    'genesis_ang_vel': '#1f77b4',
    'genesis_dof_pos': '#ff7f0e',
    'genesis_action': '#2ca02c',
    'genesis_torque': '#17becf',
    'genesis_phase': '#1f77b4',

    # Choreonoid (赤系)
    'choreonoid_ang_vel': '#d62728',
    'choreonoid_dof_pos': '#9467bd',
    'choreonoid_action': '#8c564b',
    'choreonoid_torque': '#e377c2',
    'choreonoid_phase': '#d62728',

    # No-Randomization (黒/グレー系 - 基準として目立つように)
    'norand_ang_vel': '#000000',      # 黒
    'norand_dof_pos': '#555555',      # 濃いグレー
    'norand_action': '#7f7f7f',       # グレー
    'norand_torque': '#444444',       # ダークグレー
    'norand_phase': '#000000',        # 黒
}

PLOT_CONFIG = {
    'figsize_large': (20, 15),
    'figsize_medium': (15, 12),
    'figsize_wide': (16, 12),
    'figsize_phase': (20, 15),
    'dpi': 300,
    'alpha_line': 0.7,   # 3本重なるので少し透明度を上げる
    'alpha_grid': 0.3,
    'linewidth': 2
}

# グローバル変数
OUTPUT_DIR = 'obs_comparison_plots/3way_comparison'

def load_data(genesis_path, cnoid_path, norand_path):
    """データ読み込み"""
    print("Loading data...")
    print(f"1. Genesis: {genesis_path}")
    print(f"2. Cnoid:   {cnoid_path}")
    print(f"3. NoRand:  {norand_path}")

    genesis_df = pd.read_csv(genesis_path)
    cnoid_df = pd.read_csv(cnoid_path)
    norand_df = pd.read_csv(norand_path)

    print(f"Genesis data shape: {genesis_df.shape}")
    print(f"Choreonoid data shape: {cnoid_df.shape}")
    print(f"No-Rand data shape: {norand_df.shape}")
    
    return genesis_df, cnoid_df, norand_df

def extract_obs_components(genesis_df, cnoid_df, norand_df):
    """観測値の各成分を抽出"""
    # 3つのファイルの中で最小のステップ数に合わせる
    min_steps = min(len(genesis_df), len(cnoid_df), len(norand_df))
    # min_steps = 50 # デバッグ用固定
    print(f"Analyzing {min_steps} steps")
    
    def get_data(df, prefix, count, offset=0):
        return np.array([df[f'{prefix}{i+offset}'].iloc[:min_steps] for i in range(count)]).T

    data = {}
    
    # データセットごとの処理
    sources = {
        'genesis': genesis_df,
        'cnoid': cnoid_df,
        'norand': norand_df
    }

    for name, df in sources.items():
        # Obs mapping
        data[f'{name}_ang_vel'] = get_data(df, 'obs_', 3, 0)      # obs_0~2
        data[f'{name}_dof_pos'] = get_data(df, 'obs_', 12, 9)     # obs_9~20
        data[f'{name}_dof_vel'] = get_data(df, 'obs_', 12, 21)    # obs_21~32
        data[f'{name}_actions'] = get_data(df, 'obs_', 12, 33)    # obs_33~44
        
        # Physics state
        data[f'{name}_dof_pos_full'] = get_data(df, 'dof_pos_', 12)
        data[f'{name}_dof_vel_full'] = get_data(df, 'dof_vel_', 12)
        
        # Torque (存在確認)
        if 'torque_0' in df.columns:
            data[f'{name}_torque'] = get_data(df, 'torque_', 12)
        else:
            data[f'{name}_torque'] = None

    return data

def plot_base_ang_vel_comparison(data):
    """Base Angular Velocity比較 (3者比較)"""
    steps = np.arange(len(data['genesis_ang_vel']))
    
    fig, axes = plt.subplots(1, 3, figsize=PLOT_CONFIG['figsize_wide'])
    ang_vel_names = ['Roll Rate', 'Pitch Rate', 'Yaw Rate']
    
    for i in range(3):
        # Genesis
        axes[i].plot(steps, data['genesis_ang_vel'][:, i], 
                    label='Genesis', color=COLORS['genesis_ang_vel'],
                    lw=PLOT_CONFIG['linewidth'], alpha=PLOT_CONFIG['alpha_line'])
        # Choreonoid
        axes[i].plot(steps, data['cnoid_ang_vel'][:, i], 
                    label='Choreonoid', color=COLORS['choreonoid_ang_vel'],
                    lw=PLOT_CONFIG['linewidth'], alpha=PLOT_CONFIG['alpha_line'])
        # No-Rand
        axes[i].plot(steps, data['norand_ang_vel'][:, i], 
                    label='No-Rand', color=COLORS['norand_ang_vel'],
                    lw=PLOT_CONFIG['linewidth'], alpha=PLOT_CONFIG['alpha_line'], linestyle='--')
        
        axes[i].set_title(f'{ang_vel_names[i]} (obs_{i})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Angular Velocity [rad/s]')
        axes[i].legend()
        axes[i].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
        
        # 統計情報を表示
        g_mean = np.mean(data['genesis_ang_vel'][:, i])
        c_mean = np.mean(data['cnoid_ang_vel'][:, i])
        n_mean = np.mean(data['norand_ang_vel'][:, i])
        
        axes[i].text(0.02, 0.98, f'Gen: {g_mean:.4f}\nCnoid: {c_mean:.4f}\nNoRn: {n_mean:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), fontsize=8)
    
    plt.suptitle('3-Way Comparison: Base Angular Velocity', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/base_ang_vel_comparison.png', dpi=PLOT_CONFIG['dpi'])
    print("✓ Base angular velocity comparison plot saved")

def plot_joint_metric_comparison(data, metric_name, file_suffix, ylim=None):
    """関節データの汎用プロット関数 (Pos, Vel, Action, Torque共通)"""
    steps = np.arange(len(data[f'genesis_{metric_name}']))
    
    fig, axes = plt.subplots(3, 4, figsize=PLOT_CONFIG['figsize_large'])
    axes = axes.flatten()
    
    # キー名とラベル色の設定
    keys = {
        'G': (f'genesis_{metric_name}', COLORS.get(f'genesis_{metric_name}', COLORS['genesis_dof_pos'])),
        'C': (f'cnoid_{metric_name}', COLORS.get(f'choreonoid_{metric_name}', COLORS['choreonoid_dof_pos'])),
        'N': (f'norand_{metric_name}', COLORS.get(f'norand_{metric_name}', COLORS['norand_dof_pos']))
    }

    for i in range(12):
        # Plot 3 lines
        axes[i].plot(steps, data[keys['G'][0]][:, i], label='Genesis', 
                     color=keys['G'][1], lw=2, alpha=0.7)
        axes[i].plot(steps, data[keys['C'][0]][:, i], label='Cnoid', 
                     color=keys['C'][1], lw=2, alpha=0.7)
        axes[i].plot(steps, data[keys['N'][0]][:, i], label='No-Rand', 
                     color=keys['N'][1], lw=2, alpha=0.9, linestyle=':') # No-Randは点線で強調
        
        axes[i].set_title(f'{JOINT_NAMES[i]}', fontsize=10, fontweight='bold')
        
        if i >= 8: axes[i].set_xlabel('Step')
        if i % 4 == 0: axes[i].set_ylabel('Value')
        
        if i == 0: axes[i].legend(fontsize=8) # 凡例は最初だけ
        axes[i].grid(True, alpha=0.3)
        if ylim: axes[i].set_ylim(ylim)
        
        # 統計
        vals = {k: np.mean(data[v[0]][:, i]) for k, v in keys.items()}
        axes[i].text(0.02, 0.98, f"G:{vals['G']:.2f}\nC:{vals['C']:.2f}\nN:{vals['N']:.2f}", 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), fontsize=7)

    plt.suptitle(f'3-Way Comparison: {metric_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{file_suffix}.png', dpi=PLOT_CONFIG['dpi'])
    print(f"✓ {file_suffix} plot saved")

def plot_difference_analysis(data):
    """差分分析：Genesisに対する誤差を表示"""
    steps = np.arange(len(data['genesis_dof_pos']))
    
    fig, axes = plt.subplots(3, 4, figsize=PLOT_CONFIG['figsize_large'])
    axes = axes.flatten()
    
    for i in range(12):
        # Diff Genesis vs Cnoid
        diff_gc = np.abs(data['genesis_dof_pos'][:, i] - data['cnoid_dof_pos'][:, i])
        # Diff Genesis vs NoRand
        diff_gn = np.abs(data['genesis_dof_pos'][:, i] - data['norand_dof_pos'][:, i])
        
        axes[i].plot(steps, diff_gc, label='|Gen - Cnoid|', 
                    color=COLORS['choreonoid_dof_pos'], lw=1.5, alpha=0.8)
        axes[i].plot(steps, diff_gn, label='|Gen - NoRand|', 
                    color=COLORS['norand_dof_pos'], lw=1.5, alpha=0.8, linestyle='--')
        
        axes[i].set_title(f'{JOINT_NAMES[i]} Position Diff')
        axes[i].set_yscale('log') # 対数スケールで見やすく
        axes[i].grid(True, alpha=0.3)
        if i == 0: axes[i].legend(fontsize=8)

        # Mean Error
        axes[i].text(0.02, 0.95, f'Err GC: {np.mean(diff_gc):.4f}\nErr GN: {np.mean(diff_gn):.4f}',
                     transform=axes[i].transAxes, verticalalignment='top', fontsize=7,
                     bbox=dict(facecolor='white', alpha=0.8))

    plt.suptitle('Difference Analysis (Position): Error relative to Genesis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/difference_analysis.png', dpi=PLOT_CONFIG['dpi'])
    print("✓ Difference analysis plot saved")

def plot_phase_portraits(data):
    """位相図 (3者比較)"""
    fig, axes = plt.subplots(3, 4, figsize=PLOT_CONFIG['figsize_phase'])
    axes = axes.flatten()
    
    for i in range(12):
        # Genesis
        axes[i].plot(data['genesis_dof_pos_full'][:, i], data['genesis_dof_vel_full'][:, i],
                    color=COLORS['genesis_phase'], alpha=0.6, label='Genesis')
        # Cnoid
        axes[i].plot(data['cnoid_dof_pos_full'][:, i], data['cnoid_dof_vel_full'][:, i],
                    color=COLORS['choreonoid_phase'], alpha=0.6, label='Cnoid')
        # NoRand
        axes[i].plot(data['norand_dof_pos_full'][:, i], data['norand_dof_vel_full'][:, i],
                    color=COLORS['norand_phase'], alpha=0.8, linestyle=':', lw=2, label='NoRand')

        axes[i].set_title(f'{JOINT_NAMES[i]} Phase', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        if i == 0: axes[i].legend(fontsize=8)
        axes[i].set_xlim(-1.5, 1.5)
        axes[i].set_ylim(-1.5, 1.5)

    plt.suptitle('Phase Portraits: Genesis vs Cnoid vs NoRand', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/phase_portraits.png', dpi=PLOT_CONFIG['dpi'])
    print("✓ Phase portraits saved")

def print_statistics(data):
    """3者間の統計情報出力"""
    print("\n" + "="*100)
    print("COMPREHENSIVE STATISTICS (Genesis / Cnoid / NoRand)")
    print("="*100)
    
    # 略称: G=Genesis, C=Cnoid, N=NoRand
    print(f"{'Joint':<12} | {'G_Pos':<7} {'C_Pos':<7} {'N_Pos':<7} | {'G_Vel':<7} {'C_Vel':<7} {'N_Vel':<7} | {'G_Act':<7} {'C_Act':<7} {'N_Act':<7}")
    print("-" * 100)
    
    for i in range(12):
        vals = []
        for cat in ['dof_pos', 'dof_vel', 'actions']:
            vals.append(np.mean(data[f'genesis_{cat}'][:, i]))
            vals.append(np.mean(data[f'cnoid_{cat}'][:, i]))
            vals.append(np.mean(data[f'norand_{cat}'][:, i]))
            
        print(f"{JOINT_NAMES[i]:<12} | {vals[0]:.3f}   {vals[1]:.3f}   {vals[2]:.3f}   | "
              f"{vals[3]:.3f}   {vals[4]:.3f}   {vals[5]:.3f}   | {vals[6]:.3f}   {vals[7]:.3f}   {vals[8]:.3f}")

    print("-" * 100)
    # 全体誤差 (対 NoRand)
    diff_gn_pos = np.mean(np.abs(data['genesis_dof_pos'] - data['norand_dof_pos']))
    diff_cn_pos = np.mean(np.abs(data['cnoid_dof_pos'] - data['norand_dof_pos']))
    print(f"Overall Position Diff vs NoRand: Genesis={diff_gn_pos:.5f}, Cnoid={diff_cn_pos:.5f}")

def main():
    global OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description="3-Way Observation Comparison")
    parser.add_argument("-o", "--output", type=str, default="obs_comparison_plots", help="Output dir")
    
    # ファイル引数
    parser.add_argument("-g", "--genesis", type=str, 
                        default="obs_data/genesis_friction-walking-terrain2-kp2000kd50-kpkdrand-28_ckpt100_scale1.0.csv", help="Genesis CSV path")
    parser.add_argument("-c", "--cnoid", type=str,
                        default="obs_data/cnoid_friction-walking-terrain2-kp2000kd50-kpkdrand-28_ckpt100_scale1.0_rotorInertia0.1.csv", help="Choreonoid CSV path")
    parser.add_argument("-n", "--norand", type=str,
                        default="obs_data/cnoid_friction-walking-terrain1-kp2000kd50-kpkdrand-26-norand_ckpt100_scale1.0_rotorInertia0.1.csv", help="No-Randomization CSV path")

    args = parser.parse_args()
    
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("3-WAY COMPARISON ANALYSIS START")
    print("=" * 60)
    
    # データ読み込み
    g_df, c_df, n_df = load_data(args.genesis, args.cnoid, args.norand)
    data = extract_obs_components(g_df, c_df, n_df)
    
    # プロット実行
    plot_base_ang_vel_comparison(data)
    
    # 共通プロット関数で各要素を描画
    plot_joint_metric_comparison(data, 'dof_pos', 'dof_pos_comparison', ylim=(-1.2, 1.2))
    plot_joint_metric_comparison(data, 'dof_vel', 'dof_vel_comparison', ylim=(-1.2, 1.2))
    plot_joint_metric_comparison(data, 'actions', 'action_comparison', ylim=(-1.2, 1.2))
    
    if data['genesis_torque'] is not None:
        plot_joint_metric_comparison(data, 'torque', 'torque_comparison', ylim=(-300, 600))
        
    plot_difference_analysis(data)
    plot_phase_portraits(data)
    
    print_statistics(data)
    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()


"""
python3 ishiki_Errorverifi.py \
  -o obs_comparison_3way \
  -g obs_data/genesis_ishiki-walking-no-vel_ckpt2000_simple.csv \
  -c obs_data/cnoid_ishiki-walking-no-vel_ckpt2000_simple.csv \
  -n obs_data/norand_model_data.csv
"""