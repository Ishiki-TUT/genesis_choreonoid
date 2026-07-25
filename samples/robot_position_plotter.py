import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

COLORS = {
    'trajectory': '#1f77b4',      # 青
    'start_point': '#e377c2',     # ピンク
    'end_point': '#7f7f7f',       # グレー
}

PLOT_CONFIG = {
    'dpi': 300,
}

def load_robot_data(csv_file):
    """CSVファイルからロボットデータを読み込み"""
    print(f"Loading robot data from: {csv_file}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Data shape: {df.shape}")
    
    # 必要な列の存在確認
    required_cols = ['step', 'base_pos_x', 'base_pos_y', 'base_pos_z']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
    
    return df

def plot_comparison_2d_trajectory(csv_files, labels, output_dir="robot_plots"):
    """複数のCSVファイルの2D軌道を比較プロット（DRありなし比較など）"""
    sns.set_theme(style="whitegrid", context="talk")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # 中心線
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=4, alpha=0.5, label='Center Line', zorder=1)
    
    # 比較用カラーパレット
    comparison_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # ★修正: データを一度だけ読み込んで保存
    datasets = []
    for csv_file in csv_files:
        df = load_robot_data(csv_file)
        datasets.append(df)
    
    # 各データセットをプロット
    for i, (df, label) in enumerate(zip(datasets, labels)):
        color = comparison_colors[i % len(comparison_colors)]
        
        # メイン軌道
        sns.lineplot(x=df['base_pos_y'], y=df['base_pos_x'], 
                     ax=ax, sort=False, lw=2, color=color, 
                     alpha=0.7, label=f'{label} Trajectory')
        
        # 100ステップごとの点を追加
        step_interval = 100
        step_indices = range(0, len(df), step_interval)
        
        for j, idx in enumerate(step_indices):
            if idx < len(df):
                ax.scatter(df['base_pos_y'].iloc[idx], df['base_pos_x'].iloc[idx], 
                          color=color, s=60, marker='o', alpha=0.8, 
                          edgecolor='darkred', linewidth=1, zorder=4)
                
                # ステップ数をテキストで表示（オプション）
                if j % 2 == 0:  # 2つおきに表示して重複を防ぐ
                    ax.annotate(f'{df["step"].iloc[idx]:.0f}', 
                               (df['base_pos_y'].iloc[idx], df['base_pos_x'].iloc[idx]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7, color=color)
        
        # 開始点と終了点
        ax.scatter(df['base_pos_y'].iloc[0], df['base_pos_x'].iloc[0], 
                   color=color, s=150, marker='o', 
                   edgecolor='white', linewidth=2, alpha=0.8,
                   label=f'{label} Start')
        ax.scatter(df['base_pos_y'].iloc[-1], df['base_pos_x'].iloc[-1], 
                   color=color, s=150, marker='s', 
                   edgecolor='white', linewidth=2, alpha=0.8,
                   label=f'{label} End')
    
    # 軸設定
    ax.set_xlabel('Lateral Position (Y) [m]', fontsize=14)
    ax.set_ylabel('Forward Position (X) [m]', fontsize=14)
    ax.set_title('Robot 2D Trajectory Comparison', fontsize=16, fontweight='bold')
    
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.set_ylim(-0.5, 5)
    ax.set_xlim(-3, 3)
    ax.set_aspect(0.4)
    
    # ★修正: 統計情報表示（既に読み込んだデータを使用）
    stats_text_lines = []
    for i, (df, label) in enumerate(zip(datasets, labels)):
        y_range = df['base_pos_y'].max() - df['base_pos_y'].min()
        x_range = df['base_pos_x'].max() - df['base_pos_x'].min()
        total_distance = np.sum(np.sqrt(np.diff(df['base_pos_x'])**2 + np.diff(df['base_pos_y'])**2))
        
        stats_text_lines.append(f'{label}:')
        stats_text_lines.append(f'  Lat: {y_range:.3f}m')
        stats_text_lines.append(f'  Fwd: {x_range:.3f}m')
        stats_text_lines.append(f'  Dist: {total_distance:.3f}m')
        stats_text_lines.append('')
    
    stats_text = '\n'.join(stats_text_lines)
    
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/trajectory_comparison.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Trajectory comparison plot saved")

def plot_velocity_tracking(csv_files, labels, output_dir="robot_plots"):
    """速度追従性能の時系列グラフ（修正版）"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 比較用カラーパレット
    comparison_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # データ読み込み
    datasets = []
    for csv_file in csv_files:
        df = load_robot_data(csv_file)
        
        # ★データ確認：利用可能な列を表示
        print(f"\nCSV columns for {csv_file}:")
        print([col for col in df.columns if 'vel' in col])
        print(f"Shape: {df.shape}")
        
        # ★実際のデータの最初の5行を確認
        if 'base_vel_x' in df.columns:
            print(f"base_vel_x sample: {df['base_vel_x'].head().tolist()}")
        if 'cmd_vel_x' in df.columns:
            print(f"cmd_vel_x sample: {df['cmd_vel_x'].head().tolist()}")
            
        datasets.append(df)
    
    # 各データセットをプロット
    for i, (df, label) in enumerate(zip(datasets, labels)):
        color = comparison_colors[i % len(comparison_colors)]
        
        # 時間軸（秒単位に変換, dt=0.01と仮定）
        time = df['step'] * 0.01
        
        # ★修正：列の存在確認をして適切にプロット
        # 1. X方向速度追従
        if 'cmd_vel_x' in df.columns:
            axes[0, 0].plot(time, df['cmd_vel_x'], '--', 
                            color=color, linewidth=2, alpha=0.8,
                            label=f'{label} Target X')
        else:
            print(f"Warning: cmd_vel_x not found in {label}")
            
        if 'base_vel_x' in df.columns:
            axes[0, 0].plot(time, df['base_vel_x'], '-', 
                            color=color, linewidth=1.5, alpha=0.9,
                            label=f'{label} Actual X')
        else:
            print(f"Warning: base_vel_x not found in {label}")
        
        # 2. Y方向速度追従
        if 'cmd_vel_y' in df.columns:
            axes[0, 1].plot(time, df['cmd_vel_y'], '--', 
                            color=color, linewidth=2, alpha=0.8,
                            label=f'{label} Target Y')
        else:
            print(f"Warning: cmd_vel_y not found in {label}")
            
        if 'base_vel_y' in df.columns:
            axes[0, 1].plot(time, df['base_vel_y'], '-', 
                            color=color, linewidth=1.5, alpha=0.9,
                            label=f'{label} Actual Y')
        else:
            print(f"Warning: base_vel_y not found in {label}")
        
        # 3. X方向速度誤差
        if 'base_vel_x' in df.columns and 'cmd_vel_x' in df.columns:
            x_error = df['base_vel_x'] - df['cmd_vel_x']
            axes[1, 0].plot(time, x_error, 
                            color=color, linewidth=1.5, alpha=0.8,
                            label=f'{label} X Error')
        else:
            print(f"Warning: Cannot calculate X error for {label}")
            
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Y方向速度誤差
        if 'base_vel_y' in df.columns and 'cmd_vel_y' in df.columns:
            y_error = df['base_vel_y'] - df['cmd_vel_y']
            axes[1, 1].plot(time, y_error, 
                            color=color, linewidth=1.5, alpha=0.8,
                            label=f'{label} Y Error')
        else:
            print(f"Warning: Cannot calculate Y error for {label}")
            
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 軸設定
    axes[0, 0].set_title('X Velocity Tracking', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('X Velocity [m/s]', fontsize=12)
    axes[0, 0].legend(loc='upper right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Y Velocity Tracking', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Y Velocity [m/s]', fontsize=12)
    axes[0, 1].legend(loc='upper right', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('X Velocity Error', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time [s]', fontsize=12)
    axes[1, 0].set_ylabel('X Velocity Error [m/s]', fontsize=12)
    axes[1, 0].legend(loc='upper right', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Y Velocity Error', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Time [s]', fontsize=12)
    axes[1, 1].set_ylabel('Y Velocity Error [m/s]', fontsize=12)
    axes[1, 1].legend(loc='upper right', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 全体タイトル
    fig.suptitle('Velocity Tracking Performance Comparison', fontsize=16, fontweight='bold')
    
    # ★修正：統計情報の計算と表示（列の存在確認）
    stats_text_lines = []
    for i, (df, label) in enumerate(zip(datasets, labels)):
        if 'base_vel_x' in df.columns and 'cmd_vel_x' in df.columns:
            x_mae = np.mean(np.abs(df['base_vel_x'] - df['cmd_vel_x']))
            x_rmse = np.sqrt(np.mean((df['base_vel_x'] - df['cmd_vel_x'])**2))
        else:
            x_mae = x_rmse = float('nan')
            
        if 'base_vel_y' in df.columns and 'cmd_vel_y' in df.columns:
            y_mae = np.mean(np.abs(df['base_vel_y'] - df['cmd_vel_y']))
            y_rmse = np.sqrt(np.mean((df['base_vel_y'] - df['cmd_vel_y'])**2))
        else:
            y_mae = y_rmse = float('nan')
        
        stats_text_lines.append(f'{label}:')
        stats_text_lines.append(f'  X MAE: {x_mae:.4f} m/s' if not np.isnan(x_mae) else '  X MAE: N/A')
        stats_text_lines.append(f'  Y MAE: {y_mae:.4f} m/s' if not np.isnan(y_mae) else '  Y MAE: N/A')
        stats_text_lines.append(f'  X RMSE: {x_rmse:.4f} m/s' if not np.isnan(x_rmse) else '  X RMSE: N/A')
        stats_text_lines.append(f'  Y RMSE: {y_rmse:.4f} m/s' if not np.isnan(y_rmse) else '  Y RMSE: N/A')
        stats_text_lines.append('')
    
    stats_text = '\n'.join(stats_text_lines)
    
    # 統計情報をプロット内に表示
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)  # 統計情報用にスペース確保
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/velocity_tracking_comparison.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Velocity tracking comparison plot saved")
    
    # ★修正：コンソールに統計情報出力
    print("\n" + "="*60)
    print("VELOCITY TRACKING PERFORMANCE SUMMARY")
    print("="*60)
    for i, (df, label) in enumerate(zip(datasets, labels)):
        print(f"\n{label}:")
        
        if 'base_vel_x' in df.columns and 'cmd_vel_x' in df.columns:
            x_mae = np.mean(np.abs(df['base_vel_x'] - df['cmd_vel_x']))
            x_rmse = np.sqrt(np.mean((df['base_vel_x'] - df['cmd_vel_x'])**2))
            print(f"  X Velocity - MAE: {x_mae:.4f} m/s, RMSE: {x_rmse:.4f} m/s")
        else:
            print(f"  X Velocity - データなし")
            
        if 'base_vel_y' in df.columns and 'cmd_vel_y' in df.columns:
            y_mae = np.mean(np.abs(df['base_vel_y'] - df['cmd_vel_y']))
            y_rmse = np.sqrt(np.mean((df['base_vel_y'] - df['cmd_vel_y'])**2))
            print(f"  Y Velocity - MAE: {y_mae:.4f} m/s, RMSE: {y_rmse:.4f} m/s")
        else:
            print(f"  Y Velocity - データなし")

def plot_yaw_velocity_tracking(csv_files, labels, output_dir="robot_plots"):
    """ヨー角速度追従の専用グラフ"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 比較用カラーパレット
    comparison_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # データ読み込み
    datasets = []
    for csv_file in csv_files:
        df = load_robot_data(csv_file)
        datasets.append(df)
    
    # 各データセットをプロット
    for i, (df, label) in enumerate(zip(datasets, labels)):
        color = comparison_colors[i % len(comparison_colors)]
        
        # 時間軸（秒単位に変換）
        time = df['step'] * 0.01
        
        # ヨー角速度を度/秒に変換
        cmd_yaw_deg = df['cmd_vel_yaw'] * 180 / np.pi  # rad/s → deg/s
        
        # 実際のヨー角速度計算（角度の微分）
        if 'base_yaw' in df.columns:
            actual_yaw_deg = np.gradient(df['base_yaw']) / 0.01 * 180 / np.pi
        else:
            # ヨー角がない場合は位置から推定
            actual_yaw_deg = np.zeros_like(cmd_yaw_deg)
        
        # 1. ヨー角速度追従
        axes[0].plot(time, cmd_yaw_deg, '--', 
                     color=color, linewidth=2, alpha=0.8,
                     label=f'{label} Target Yaw Rate')
        axes[0].plot(time, actual_yaw_deg, '-', 
                     color=color, linewidth=1.5, alpha=0.9,
                     label=f'{label} Actual Yaw Rate')
        
        # 2. ヨー角速度誤差
        yaw_error = actual_yaw_deg - cmd_yaw_deg
        axes[1].plot(time, yaw_error, 
                     color=color, linewidth=1.5, alpha=0.8,
                     label=f'{label} Yaw Error')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 軸設定
    axes[0].set_title('Yaw Rate Tracking', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Yaw Rate [deg/s]', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Yaw Rate Error', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time [s]', fontsize=12)
    axes[1].set_ylabel('Yaw Rate Error [deg/s]', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Yaw Rate Tracking Performance', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/yaw_tracking_comparison.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Yaw tracking comparison plot saved")

# ベース位置と速度の包括的比較
def plot_comprehensive_base_comparison(csv_files, labels, output_dir="robot_plots"):
    """ベース位置と速度の包括的比較プロット"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 比較用カラーパレット
    comparison_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # データ読み込み
    datasets = []
    for csv_file in csv_files:
        df = load_robot_data(csv_file)
        datasets.append(df)
    
    # 各データセットをプロット
    for i, (df, label) in enumerate(zip(datasets, labels)):
        color = comparison_colors[i % len(comparison_colors)]
        
        # 1. ベース位置
        axes[0, 0].plot(df['step'], df['base_pos_x'], label=f'{label} X', linewidth=2, color=color)
        axes[0, 1].plot(df['step'], df['base_pos_y'], label=f'{label} Y', linewidth=2, color=color)
        axes[0, 2].plot(df['step'], df['base_pos_z'], label=f'{label} Z', linewidth=2, color=color)
        
        # 2. ベース速度
        axes[1, 0].plot(df['step'], df['base_vel_x'], label=f'{label} X', linewidth=2, color=color)
        axes[1, 1].plot(df['step'], df['base_vel_y'], label=f'{label} Y', linewidth=2, color=color)
        axes[1, 2].plot(df['step'], df['base_vel_z'], label=f'{label} Z', linewidth=2, color=color)
    
    # 軸設定
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel('Step')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    axes[0, 0].set_ylabel('Position X [m]', fontsize=12)
    axes[0, 1].set_ylabel('Position Y [m]', fontsize=12)
    axes[0, 2].set_ylabel('Position Z [m]', fontsize=12)
    
    axes[1, 0].set_ylabel('Velocity X [m/s]', fontsize=12)
    axes[1, 1].set_ylabel('Velocity Y [m/s]', fontsize=12)
    axes[1, 2].set_ylabel('Velocity Z [m/s]', fontsize=12)
    
    # 全体タイトル
    fig.suptitle('Comprehensive Base Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/comprehensive_base_comparison.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print("✓ Comprehensive base comparison plot saved")

def main():
    # ★直接CSVファイルパスを指定
    csv_files = [
        'obs_data/cnoid_friction-walking-terrain1-kp2000kd50-kpkdrand-26-norand_ckpt500_scale1.0_position.csv',
        'obs_data/cnoid_friction-walking-terrain2-kp2000kd50-kpkdrand-28_ckpt500_scale1.0_position.csv'
    ]
    
    # ラベル設定
    labels = [
        'DR-disabled',
        'DR-enabled'
    ]
    
    # 出力ディレクトリ
    output_dir = "robot_plots"
    
    # プロット生成設定
    generate_trajectory = True     # 軌道比較
    generate_velocity = True       # 速度追従
    generate_yaw = True           # ヨー角速度追従
    generate_comprehensive = True  # ベース位置と速度の包括的比較
    
    # ====== 実行部分 ======
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # ファイル存在確認
    valid_files = []
    valid_labels = []
    for csv_file, label in zip(csv_files, labels):
        if os.path.exists(csv_file):
            valid_files.append(csv_file)
            valid_labels.append(label)
        else:
            print(f"Warning: CSV file not found: {csv_file}")
    
    if len(valid_files) == 0:
        print("No valid CSV files found!")
        return
    
    # プロット生成
    print(f"\n=== Analysis: {len(valid_files)} datasets ===")
    for csv_file, label in zip(valid_files, valid_labels):
        print(f"  {label}: {csv_file}")
    
    if generate_trajectory:
        plot_comparison_2d_trajectory(valid_files, valid_labels, output_dir)
    
    if generate_velocity and len(valid_files) >= 1:
        plot_velocity_tracking(valid_files, valid_labels, output_dir)
    
    if generate_yaw and len(valid_files) >= 1:
        plot_yaw_velocity_tracking(valid_files, valid_labels, output_dir)
    
    if generate_comprehensive and len(valid_files) >= 1:
        plot_comprehensive_base_comparison(valid_files, valid_labels, output_dir)
    
    print(f"\n✓ All plots saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()