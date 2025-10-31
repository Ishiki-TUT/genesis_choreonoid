import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 共通設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

# 定数定義
JOINT_NAMES = ['R_HIP_Y', 'R_HIP_R', 'R_HIP_P', 'R_KNEE', 'R_ANKLE_P', 'R_ANKLE_R',
               'L_HIP_Y', 'L_HIP_R', 'L_HIP_P', 'L_KNEE', 'L_ANKLE_P', 'L_ANKLE_R']

COLORS = {
    'genesis': 'blue',
    'genesis_light': 'lightblue',
    'genesis_dark': 'darkblue',
    'choreonoid': 'red',
    'choreonoid_light': 'lightcoral',
    'choreonoid_dark': 'darkred'
}

PLOT_CONFIG = {
    'figsize_large': (20, 15),
    'figsize_medium': (15, 12),
    'figsize_wide': (16, 12),
    'dpi': 300,
    'alpha_line': 0.8,
    'alpha_grid': 0.3,
    'linewidth': 2,
    'linewidth_thick': 3
}

# 結果保存用ディレクトリ作成
os.makedirs('comparison_plots', exist_ok=True)

def load_data():
    """データ読み込み"""
    genesis_df = pd.read_csv('obs_data/genesis_ishiki-walking-no-vel_ckpt2000_simple.csv')
    cnoid_df = pd.read_csv('obs_data/cnoid_ishiki-walking-no-vel_ckpt2000_simple.csv')
    return genesis_df, cnoid_df

def calc_action_obs_diff(df):
    """
    アクションと対応する観測値の差分を計算
    観測値の構造:
    - obs_0~2: base_ang_vel (3)
    - obs_3~5: projected_gravity (3) 
    - obs_6~8: commands (3)
    - obs_9~20: dof_pos - default_dof_pos (12)
    - obs_21~32: dof_vel (12)
    - obs_33~44: actions (12) ← これがactionに対応
    """
    action_cols = [col for col in df.columns if col.startswith('action_')]
    num_actions = len(action_cols)
    obs_action_start_idx = 33  # obs_33からがactions
    
    diffs = []
    for i in range(num_actions):
        action_col = f'action_{i}'
        obs_col = f'obs_{obs_action_start_idx + i}'
        
        if action_col in df.columns and obs_col in df.columns:
            diff = np.abs(df[action_col] - df[obs_col])
            diffs.append(diff)
        else:
            print(f"Warning: {action_col} or {obs_col} not found")
    
    return np.array(diffs).T

def get_common_data():
    """共通データを取得"""
    genesis_df, cnoid_df = load_data()
    genesis_diffs = calc_action_obs_diff(genesis_df)
    cnoid_diffs = calc_action_obs_diff(cnoid_df)
    return genesis_diffs, cnoid_diffs

def plot_1_time_series():
    """1. 時系列プロット"""
    genesis_diffs, cnoid_diffs = get_common_data()
    
    fig, axes = plt.subplots(3, 4, figsize=PLOT_CONFIG['figsize_large'])
    axes = axes.flatten()
    
    for i in range(12):
        axes[i].plot(genesis_diffs[:, i], label='Genesis', 
                    alpha=PLOT_CONFIG['alpha_line'], 
                    linewidth=PLOT_CONFIG['linewidth'], 
                    color=COLORS['genesis'])
        axes[i].plot(cnoid_diffs[:, i], label='Choreonoid', 
                    alpha=PLOT_CONFIG['alpha_line'], 
                    linewidth=PLOT_CONFIG['linewidth'], 
                    color=COLORS['choreonoid'])
        axes[i].set_title(f'{JOINT_NAMES[i]} Error', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('|Action - Obs|')
        axes[i].legend()
        axes[i].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
        
        # Y軸の範囲を調整
        max_val = max(np.max(genesis_diffs[:, i]), np.max(cnoid_diffs[:, i]))
        axes[i].set_ylim(0, max_val * 1.1)
    
    plt.suptitle('Action-Observation Error Time Series Comparison\n(Action vs obs_33~44)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_plots/1_time_series_comparison.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("1. Time series plot saved as '1_time_series_comparison.png'")

def plot_2_box_plots():
    """2. 統計サマリー比較（箱ひげ図）"""
    genesis_diffs, cnoid_diffs = get_common_data()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PLOT_CONFIG['figsize_medium'])
    
    # Genesis
    genesis_box_data = [genesis_diffs[:, i] for i in range(12)]
    bp1 = ax1.boxplot(genesis_box_data, labels=JOINT_NAMES, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS['genesis_light'])
    ax1.set_title('Genesis: Action-Observation Error Distribution\n(Action vs obs_33~44)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('|Action - Obs|')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=PLOT_CONFIG['alpha_grid'])
    
    # Choreonoid
    cnoid_box_data = [cnoid_diffs[:, i] for i in range(12)]
    bp2 = ax2.boxplot(cnoid_box_data, labels=JOINT_NAMES, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS['choreonoid_light'])
    ax2.set_title('Choreonoid: Action-Observation Error Distribution\n(Action vs obs_33~44)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('|Action - Obs|')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=PLOT_CONFIG['alpha_grid'])
    
    plt.tight_layout()
    plt.savefig('comparison_plots/2_box_plot_comparison.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("2. Box plot comparison saved as '2_box_plot_comparison.png'")

def plot_3_heatmap():
    """3. ヒートマップ比較"""
    genesis_diffs, cnoid_diffs = get_common_data()
    
    # 統計値を計算
    stats = {
        'genesis': {
            'mean': np.array([np.mean(genesis_diffs[:, i]) for i in range(12)]),
            'std': np.array([np.std(genesis_diffs[:, i]) for i in range(12)]),
            'max': np.array([np.max(genesis_diffs[:, i]) for i in range(12)])
        },
        'cnoid': {
            'mean': np.array([np.mean(cnoid_diffs[:, i]) for i in range(12)]),
            'std': np.array([np.std(cnoid_diffs[:, i]) for i in range(12)]),
            'max': np.array([np.max(cnoid_diffs[:, i]) for i in range(12)])
        }
    }
    
    fig, axes = plt.subplots(3, 1, figsize=PLOT_CONFIG['figsize_medium'])
    
    # ヒートマップの設定
    heatmap_configs = [
        {'data': np.vstack([stats['genesis']['mean'], stats['cnoid']['mean']]),
         'title': 'Mean Action-Observation Error Comparison\n(Action vs obs_33~44)',
         'cmap': 'viridis'},
        {'data': np.vstack([stats['genesis']['std'], stats['cnoid']['std']]),
         'title': 'Standard Deviation Comparison',
         'cmap': 'plasma'},
        {'data': np.vstack([stats['genesis']['max'], stats['cnoid']['max']]),
         'title': 'Maximum Error Comparison',
         'cmap': 'inferno'}
    ]
    
    for i, config in enumerate(heatmap_configs):
        sns.heatmap(config['data'], 
                    xticklabels=JOINT_NAMES,
                    yticklabels=['Genesis', 'Choreonoid'],
                    annot=True, fmt='.4f', cmap=config['cmap'], ax=axes[i])
        axes[i].set_title(config['title'], fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comparison_plots/3_heatmap_comparison.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("3. Heatmap comparison saved as '3_heatmap_comparison.png'")

def calculate_leg_errors(diffs):
    """右脚・左脚のエラーを計算"""
    right_leg = np.sum(diffs[:, :6], axis=1)  # R_HIP_Y to R_ANKLE_R
    left_leg = np.sum(diffs[:, 6:], axis=1)   # L_HIP_Y to L_ANKLE_R
    return right_leg, left_leg

def plot_4_cumulative_error():
    """4. 累積誤差プロット"""
    genesis_diffs, cnoid_diffs = get_common_data()
    
    # 各種エラーを計算
    genesis_total_error = np.sum(genesis_diffs, axis=1)
    cnoid_total_error = np.sum(cnoid_diffs, axis=1)
    
    genesis_right_leg, genesis_left_leg = calculate_leg_errors(genesis_diffs)
    cnoid_right_leg, cnoid_left_leg = calculate_leg_errors(cnoid_diffs)
    
    fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG['figsize_wide'])
    
    # 1. 全actionの累積誤差
    axes[0, 0].plot(np.cumsum(genesis_total_error), label='Genesis', 
                    linewidth=PLOT_CONFIG['linewidth_thick'], color=COLORS['genesis'])
    axes[0, 0].plot(np.cumsum(cnoid_total_error), label='Choreonoid', 
                    linewidth=PLOT_CONFIG['linewidth_thick'], color=COLORS['choreonoid'])
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Cumulative Total Error')
    axes[0, 0].set_title('Cumulative Total Error Over Time\n(Action vs obs_33~44)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
    
    # 2. 平時誤差
    axes[0, 1].plot(genesis_total_error, label='Genesis', alpha=0.7, color=COLORS['genesis'])
    axes[0, 1].plot(cnoid_total_error, label='Choreonoid', alpha=0.7, color=COLORS['choreonoid'])
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Total Error per Step')
    axes[0, 1].set_title('Total Error per Time Step', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
    
    # 3. 右脚vs左脚の累積誤差
    leg_plot_config = [
        {'data': np.cumsum(genesis_right_leg), 'label': 'Genesis Right Leg', 
         'style': '--', 'color': COLORS['genesis']},
        {'data': np.cumsum(genesis_left_leg), 'label': 'Genesis Left Leg', 
         'style': '-', 'color': COLORS['genesis']},
        {'data': np.cumsum(cnoid_right_leg), 'label': 'Choreonoid Right Leg', 
         'style': '--', 'color': COLORS['choreonoid']},
        {'data': np.cumsum(cnoid_left_leg), 'label': 'Choreonoid Left Leg', 
         'style': '-', 'color': COLORS['choreonoid']}
    ]
    
    for config in leg_plot_config:
        axes[1, 0].plot(config['data'], label=config['label'], 
                       linestyle=config['style'], color=config['color'])
    
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Cumulative Error')
    axes[1, 0].set_title('Cumulative Error: Right vs Left Leg', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
    
    # 4. 統計サマリー
    stats_data = {
        'Genesis Total': [np.mean(genesis_total_error), np.std(genesis_total_error)],
        'Choreonoid Total': [np.mean(cnoid_total_error), np.std(cnoid_total_error)],
        'Genesis Right': [np.mean(genesis_right_leg), np.std(genesis_right_leg)],
        'Choreonoid Right': [np.mean(cnoid_right_leg), np.std(cnoid_right_leg)],
        'Genesis Left': [np.mean(genesis_left_leg), np.std(genesis_left_leg)],
        'Choreonoid Left': [np.mean(cnoid_left_leg), np.std(cnoid_left_leg)]
    }
    
    x_pos = np.arange(len(stats_data))
    means = [stats_data[key][0] for key in stats_data.keys()]
    stds = [stats_data[key][1] for key in stats_data.keys()]
    
    bar_colors = [COLORS['genesis_light'], COLORS['choreonoid_light'], 
                  COLORS['genesis'], COLORS['choreonoid'], 
                  COLORS['genesis_dark'], COLORS['choreonoid_dark']]
    
    bars = axes[1, 1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=bar_colors)
    axes[1, 1].set_xlabel('Comparison Category')
    axes[1, 1].set_ylabel('Mean Error ± Std')
    axes[1, 1].set_title('Error Statistics Summary', fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(stats_data.keys(), rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=PLOT_CONFIG['alpha_grid'])
    
    plt.tight_layout()
    plt.savefig('comparison_plots/4_cumulative_error_analysis.png', 
                dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("4. Cumulative error analysis saved as '4_cumulative_error_analysis.png'")

def print_summary_statistics():
    """統計サマリーを出力"""
    genesis_diffs, cnoid_diffs = get_common_data()
    
    print("\n" + "="*90)
    print("ACTION-OBSERVATION ERROR ANALYSIS (Action vs obs_33~44)")
    print("="*90)
    
    print(f"{'Joint':<12} {'Genesis Mean':<13} {'Genesis Std':<12} {'Choreonoid Mean':<15} {'Choreonoid Std':<14} {'Difference':<10}")
    print("-"*90)
    
    for i in range(12):
        g_mean = np.mean(genesis_diffs[:, i])
        g_std = np.std(genesis_diffs[:, i])
        c_mean = np.mean(cnoid_diffs[:, i])
        c_std = np.std(cnoid_diffs[:, i])
        diff = g_mean - c_mean
        
        print(f"{JOINT_NAMES[i]:<12} {g_mean:<13.6f} {g_std:<12.6f} {c_mean:<15.6f} {c_std:<14.6f} {diff:<10.6f}")
    
    # 全体統計
    genesis_total = np.sum(genesis_diffs, axis=1)
    cnoid_total = np.sum(cnoid_diffs, axis=1)
    
    print("\n" + "-"*90)
    print("OVERALL STATISTICS")
    print("-"*90)
    print(f"Genesis Total Error:     Mean={np.mean(genesis_total):.6f}, Std={np.std(genesis_total):.6f}")
    print(f"Choreonoid Total Error:  Mean={np.mean(cnoid_total):.6f}, Std={np.std(cnoid_total):.6f}")
    print(f"Difference (G-C):        {np.mean(genesis_total) - np.mean(cnoid_total):.6f}")
    
    # データ構造の確認情報
    print("\n" + "-"*90)
    print("DATA VERIFICATION")
    print("-"*90)
    print("Observation structure:")
    print("- obs_0~2:   base_ang_vel (3)")
    print("- obs_3~5:   projected_gravity (3)")
    print("- obs_6~8:   commands (3)")
    print("- obs_9~20:  dof_pos - default_dof_pos (12)")
    print("- obs_21~32: dof_vel (12)")
    print("- obs_33~44: actions (12) ← Compared with action_0~11")

def main():
    """メイン実行関数"""
    print("Action-Observation Error Analysis Starting...")
    print("Comparing action_0~11 with obs_33~44 (actions in observation)")
    print("="*70)
    
    # 4つの比較プロットを生成
    plot_functions = [plot_1_time_series, plot_2_box_plots, plot_3_heatmap, plot_4_cumulative_error]
    
    for plot_func in plot_functions:
        plot_func()
    
    # 統計サマリーを出力
    print_summary_statistics()
    
    print("\n" + "="*70)
    print("All plots saved in 'comparison_plots/' directory:")
    print("1. 1_time_series_comparison.png")
    print("2. 2_box_plot_comparison.png")
    print("3. 3_heatmap_comparison.png")
    print("4. 4_cumulative_error_analysis.png")
    print("="*70)

if __name__ == "__main__":
    main()