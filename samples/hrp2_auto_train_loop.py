import os
import sys
import json
import argparse
from datetime import datetime

# パス設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "irsl_rl"))

import subprocess
import torch
import numpy as np
import pickle

class RewardOptimizer:
    def __init__(self, base_exp_name="hrp2-walking", max_iterations=1000, num_loops=5):
        self.base_exp_name = base_exp_name
        self.max_iterations = max_iterations
        self.num_loops = num_loops
        self.reward_history = []
        self.best_reward_config = None
        self.best_score = float('-inf')
        
        # リワード設定（初期値）
        # ★重要: 片足立ちで止まる問題を解決するため、forward velocityを最優先
        self.reward_scales = {
            "tracking_lin_vel": 5.0,          # ★ 大幅増加: 前進を最優先 (was 1.0)
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "feet_air_time": 1.0,             # ★ 減少: 片足立ちより前進を優先 (was 2.0)
            "feet_alternating_pos": 1.5,      # ★ 減少: 前進が十分な時に強化 (was 3.0)
            "ankle_regularization": -0.5,
            "orientation": 1.0,
        }
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"auto_train_logs/{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
    def run_training(self, loop_num):
        """トレーニングを実行"""
        exp_name = f"{self.base_exp_name}_loop{loop_num}"
        print(f"\n{'='*60}")
        print(f"Loop {loop_num}: Training with updated rewards")
        print(f"{'='*60}")
        
        # hrp2_train.py を実行
        cmd = [
            "python3", "hrp2_train.py",
            "-e", exp_name,
            "--max_iterations", str(self.max_iterations)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
        
        return result.returncode == 0, exp_name
    
    def run_evaluation(self, exp_name, num_episodes=5):
        """評価を実行して歩行スコアを計算"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")
        
        cmd = [
            "python3", "hrp2_eval_gs.py",
            "-e", exp_name,
            "--ckpt", "100"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
        
        # 評価スコアを計算（仮実装）
        # 実際は hrp2_eval_gs.py から動作メトリクスを取得する必要があります
        score = self.calculate_walk_score(exp_name)
        
        return score
    
    def calculate_walk_score(self, exp_name):
        """
        観測データから歩行スコアを計算
        目標：足を完全に離して（滞空時間）、左右交互に歩行
        """
        # メトリクスを取得（評価ログから）
        metrics = self.get_evaluation_metrics(exp_name)
        
        if metrics is None or len(metrics) == 0:
            print(f"Warning: Could not get metrics for {exp_name}")
            return 0.0
        
        try:
            # 各スコアを計算
            air_time_score = self.calc_air_time_score(metrics)
            alternating_score = self.calc_alternating_score(metrics)
            speed_stability_score = self.calc_speed_stability(metrics)
            symmetry_score = self.calc_gait_symmetry(metrics)
            
            # 加重平均で総合スコアを計算
            total_score = (
                0.35 * air_time_score +        # 足の滞空時間（最重要）
                0.35 * alternating_score +     # 左右交互性（最重要）
                0.20 * speed_stability_score + # 速度安定性
                0.10 * symmetry_score          # 歩行対称性
            )
            
            print(f"\n{'─'*60}")
            print(f"Walk Quality Metrics for {exp_name}:")
            print(f"{'─'*60}")
            print(f"Air Time Score:          {air_time_score:.4f} (足の滞空時間)")
            print(f"Alternating Score:       {alternating_score:.4f} (足の交互性)")
            print(f"Speed Stability Score:   {speed_stability_score:.4f} (速度安定性)")
            print(f"Symmetry Score:          {symmetry_score:.4f} (歩行対称性)")
            print(f"{'─'*60}")
            print(f"TOTAL SCORE:             {total_score:.4f}")
            print(f"{'─'*60}\n")
            
            return min(total_score, 1.0)
            
        except Exception as e:
            print(f"Error calculating walk score: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def get_evaluation_metrics(self, exp_name):
        """評価ログからメトリクスを取得"""
        log_file = os.path.join("logs", exp_name, "eval_metrics.pkl")
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'rb') as f:
                    metrics = pickle.load(f)
                
                # メトリクスを統合テンソルに変換
                converted_metrics = {}
                for key, value_list in metrics.items():
                    if isinstance(value_list, list) and len(value_list) > 0:
                        if isinstance(value_list[0], torch.Tensor):
                            # テンソルを結合
                            converted_metrics[key] = torch.cat(value_list).cpu().numpy()
                        else:
                            # すでに配列またはスカラー
                            converted_metrics[key] = np.array(value_list)
                    else:
                        converted_metrics[key] = np.array(value_list) if value_list else np.array([])
                
                print(f"Loaded metrics from {log_file}")
                for key, value in converted_metrics.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                
                return converted_metrics
            except Exception as e:
                print(f"Error loading metrics: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print(f"Metrics file not found: {log_file}")
            return None
    
    def calc_air_time_score(self, metrics):
        """
        足の滞空時間スコアを計算
        目標：足が十分に地面から離れている（すり足でない）
        
        滞空時間の判定基準：
        - 足の高さが 0.01m 以上 → 地面から離れている
        """
        if 'left_foot_z' not in metrics or 'right_foot_z' not in metrics:
            print("  ⚠ No foot height data available")
            return 0.0
        
        left_z = metrics['left_foot_z']   # (num_steps,)
        right_z = metrics['right_foot_z']  # (num_steps,)
        
        # テンソルの場合と配列の場合に対応
        if isinstance(left_z, torch.Tensor):
            left_z = left_z.cpu().numpy()
        if isinstance(right_z, torch.Tensor):
            right_z = right_z.cpu().numpy()
        
        # 足が地面から浮いている時間の割合
        # 足の高さが 0.01m 以上なら浮いていると判定
        left_airborne = (left_z > 0.01).astype(float)
        right_airborne = (right_z > 0.01).astype(float)
        
        left_air_ratio = left_airborne.mean()
        right_air_ratio = right_airborne.mean()
        
        # 平均滞空比
        mean_air_ratio = (left_air_ratio + right_air_ratio) / 2.0
        
        # スコア化：
        # 10% → 0.1, 30% → 0.5, 50% → 0.8, 70% → 1.0
        if mean_air_ratio < 0.1:
            air_time_score = mean_air_ratio  # 0.0-0.1
        elif mean_air_ratio < 0.5:
            air_time_score = 0.1 + (mean_air_ratio - 0.1) * 1.3  # 0.1-0.62
        else:
            air_time_score = min(mean_air_ratio / 0.7, 1.0)
        
        print(f"  📊 Left foot air ratio: {left_air_ratio:.2%} (height > 0.01m)")
        print(f"  📊 Right foot air ratio: {right_air_ratio:.2%}")
        print(f"  📊 Mean air time: {mean_air_ratio:.2%}")
        
        return float(air_time_score)
    
    def calc_alternating_score(self, metrics):
        """
        左右の足の交互性スコアを計算
        目標：左足と右足が交互に出ている（シャッフルしない）
        
        評価方法：
        1. 両足が同時に接地している時間の割合を計算
           → 0に近いほど交互性が高い
        2. 足の接地パターンの変化から交互性を評価
        3. 足の切り替え周期が規則的かチェック
        """
        if 'left_foot_z' not in metrics or 'right_foot_z' not in metrics:
            print("  ⚠ No foot height data available")
            return 0.0
        
        left_z = metrics['left_foot_z']   # (num_steps,)
        right_z = metrics['right_foot_z']  # (num_steps,)
        
        # テンソルの場合と配列の場合に対応
        if isinstance(left_z, torch.Tensor):
            left_z = left_z.cpu().numpy()
        if isinstance(right_z, torch.Tensor):
            right_z = right_z.cpu().numpy()
        
        # 足が接地している状態を判定（高さ <= 0.01m）
        left_contact = (left_z <= 0.01).astype(float)
        right_contact = (right_z <= 0.01).astype(float)
        
        # 両足が同時に接地している時間の割合
        both_contact = left_contact * right_contact
        simultaneous_ratio = both_contact.mean()
        
        # シャッフル検出：両足が同時に接地し続ける
        # 連続して両足接地が続く期間をカウント
        shuffle_periods = self._detect_shuffle_periods(both_contact)
        shuffle_penalty = len(shuffle_periods) / max(len(both_contact) / 10, 1)  # 規格化
        shuffle_penalty = min(shuffle_penalty, 1.0)
        
        # 足の切り替え周期の規則性をチェック
        left_transitions = self._detect_transitions(left_contact)
        right_transitions = self._detect_transitions(right_contact)
        
        # 交互性を検証
        alternating_count = self._count_alternating_pattern(left_transitions, right_transitions)
        max_transitions = max(len(left_transitions), len(right_transitions))
        
        if max_transitions > 0:
            alternating_ratio = alternating_count / max_transitions
        else:
            alternating_ratio = 0.0
        
        # 最終スコア = 交互性 × (1 - シャッフルペナルティ) × (1 - 同時接地)
        alternating_score = alternating_ratio * (1.0 - shuffle_penalty) * (1.0 - simultaneous_ratio * 0.5)
        
        print(f"  👣 Simultaneous contact ratio: {simultaneous_ratio:.2%} (低いほど良い)")
        print(f"  🔄 Alternating pattern ratio: {alternating_ratio:.2%}")
        print(f"  🚶 Shuffle penalty: {shuffle_penalty:.2%}")
        print(f"  → Alternating score: {alternating_score:.4f}")
        
        return min(float(alternating_score), 1.0)
    
    def _detect_transitions(self, contact_signal):
        """
        接地状態の遷移（0→1 または 1→0）を検出
        Returns: 遷移が起きたステップのリスト
        """
        transitions = []
        for i in range(1, len(contact_signal)):
            if contact_signal[i] != contact_signal[i-1]:
                transitions.append(i)
        return transitions
    
    def _detect_shuffle_periods(self, both_contact):
        """
        両足が同時に接地し続ける期間を検出
        シャッフル = 両足が同時に接地し続ける状態
        """
        shuffle_periods = []
        duration = 0
        for in_contact in both_contact:
            if in_contact > 0.5:  # 両足接地
                duration += 1
            else:
                if duration > 5:  # 5ステップ以上の連続接地
                    shuffle_periods.append(duration)
                duration = 0
        if duration > 5:
            shuffle_periods.append(duration)
        return shuffle_periods
    
    def _count_alternating_pattern(self, left_trans, right_trans):
        """
        左右の足が交互に切り替わっているかカウント
        """
        if not left_trans or not right_trans:
            return 0
        
        alternating = 0
        for i in range(min(len(left_trans), len(right_trans)) - 1):
            # 左の次のイベントが右のイベントより後か、またはその逆
            if i < len(left_trans) - 1 and i < len(right_trans):
                if (left_trans[i] < right_trans[i] < left_trans[i + 1]) or \
                   (right_trans[i] < left_trans[i] < right_trans[i + 1]):
                    alternating += 1
        return alternating
    
    def calc_speed_stability(self, metrics):
        """
        歩行速度の安定性スコアを計算
        
        評価方法：
        1. 指令速度への追従度（誤差が小さいほど良い）
        2. 速度の安定性（ばらつきが小さいほど良い）
        """
        if 'lin_vel' not in metrics or 'command_vel' not in metrics:
            print("  ⚠ No velocity data available")
            return 0.0
        
        velocities = metrics['lin_vel']        # (num_steps,)
        commands = metrics['command_vel']       # (num_steps,)
        
        # テンソルの場合と配列の場合に対応
        if isinstance(velocities, torch.Tensor):
            velocities = velocities.cpu().numpy()
        if isinstance(commands, torch.Tensor):
            commands = commands.cpu().numpy()
        
        # 速度誤差
        vel_error = np.abs(velocities - commands)
        mean_error = vel_error.mean()
        
        # 速度のばらつき（安定性）
        if len(velocities) > 1:
            vel_std = velocities.std()
            vel_mean = velocities.mean()
            
            if vel_mean != 0:
                smoothness = 1.0 - min(vel_std / abs(vel_mean) / 0.5, 1.0)
            else:
                smoothness = 0.0
        else:
            smoothness = 0.5
        
        # 目標速度への追従度
        # 平均誤差が小さいほど良い
        if commands.mean() > 0:
            tracking_score = max(1.0 - (mean_error / commands.mean()), 0.0)
        else:
            tracking_score = 0.5
        
        # 最終スコア
        speed_stability = (tracking_score * 0.6 + smoothness * 0.4)
        
        print(f"  🎯 Velocity tracking error: {mean_error:.4f} m/s")
        print(f"  📈 Velocity smoothness: {smoothness:.4f}")
        print(f"  ⚖️  Tracking score: {tracking_score:.4f}")
        print(f"  → Speed stability: {speed_stability:.4f}")
        
        return float(speed_stability)
    
    def calc_gait_symmetry(self, metrics):
        """
        歩行の左右対称性スコアを計算
        
        評価方法：
        1. 左右の足の高さの差を計算
        2. 浮遊時間の左右差を計算
        3. 運動の周期性をチェック
        """
        if 'left_foot_z' not in metrics or 'right_foot_z' not in metrics:
            print("  ⚠ No foot height data available")
            return 0.0
        
        left_z = metrics['left_foot_z']
        right_z = metrics['right_foot_z']
        
        # テンソルの場合と配列の場合に対応
        if isinstance(left_z, torch.Tensor):
            left_z = left_z.cpu().numpy()
        if isinstance(right_z, torch.Tensor):
            right_z = right_z.cpu().numpy()
        
        # 左右の足の高さの差
        z_diff = np.abs(left_z - right_z)
        mean_z_diff = z_diff.mean()
        
        # 差が小さいほど対称的
        height_symmetry = max(1.0 - mean_z_diff / 0.05, 0.0)
        
        # 滞空時間の左右差を計算
        left_air_time = (left_z > 0.01).sum()
        right_air_time = (right_z > 0.01).sum()
        
        total_air = left_air_time + right_air_time
        if total_air > 0:
            air_balance = 1.0 - abs(left_air_time - right_air_time) / total_air
        else:
            air_balance = 0.0
        
        # 最終スコア
        symmetry_score = (height_symmetry * 0.6 + air_balance * 0.4)
        
        print(f"  🔄 Left-Right height difference: {mean_z_diff:.4f} m")
        print(f"  ⚖️  Air time balance: {air_balance:.2%}")
        print(f"  → Symmetry score: {symmetry_score:.4f}")
        
        return float(symmetry_score)
    
    def adjust_rewards(self, loop_num, score):
        """スコアに基づいてリワードを調整"""
        print(f"\n{'='*60}")
        print(f"Adjusting rewards based on score: {score:.4f}")
        print(f"{'='*60}")
        
        # 段階的な調整戦略（前進速度を最優先とする）
        if score < 0.2:
            # 非常に低スコア：ロボットが動いていない可能性
            print("  Action: Model is not moving - Increase forward velocity reward")
            self.reward_scales["tracking_lin_vel"] *= 1.3    # 大幅増加
            self.reward_scales["feet_air_time"] *= 0.8       # 減少（片足立ちの可能性）
            self.reward_scales["feet_alternating_pos"] *= 0.9
            
        elif score < 0.4:
            # 低スコア：片足立ちまたは遅い動き
            print("  Action: Model is stationary or moving slowly")
            self.reward_scales["tracking_lin_vel"] *= 1.2    # 前進を強化
            self.reward_scales["feet_air_time"] *= 0.9       # 滞空時間を減らす
            self.reward_scales["feet_alternating_pos"] *= 1.1
            
        elif score < 0.6:
            # 中程度：基本的な歩行はできている
            print("  Action: Basic walking achieved - Improving gait pattern")
            self.reward_scales["feet_alternating_pos"] *= 1.2  # 交互性を強化
            self.reward_scales["ankle_regularization"] *= 1.1
            
        elif score < 0.8:
            # 良好：交互歩行が形成されている
            print("  Action: Good gait pattern - Fine-tuning stability")
            self.reward_scales["orientation"] *= 1.05
            self.reward_scales["action_rate"] *= 0.95  # 平滑化
            
        else:
            # 優秀：高品質な歩行
            print("  Action: Excellent gait - Pursuing precision")
            self.reward_scales["ankle_regularization"] *= 1.15
            self.reward_scales["orientation"] *= 1.1
        
        # リワード値が極端にならないようにクリップ
        for key in self.reward_scales:
            self.reward_scales[key] = max(min(self.reward_scales[key], 15.0), -15.0)
        
        self.print_reward_config()
    
    def print_reward_config(self):
        """現在のリワード設定を表示"""
        print("\nCurrent reward configuration:")
        for key, value in self.reward_scales.items():
            print(f"  {key}: {value:.4f}")
    
    def save_reward_config(self, loop_num):
        """リワード設定を保存"""
        config_file = os.path.join(self.log_dir, f"rewards_loop{loop_num}.json")
        with open(config_file, 'w') as f:
            json.dump(self.reward_scales, f, indent=2)
        print(f"Saved reward config to {config_file}")
    
    def update_train_config(self):
        """hrp2_train.py の設定を更新"""
        # 実装例：hrp2_train.py のリワード設定を動的に更新
        # ここでは設定ファイルに書き込むか、環境変数で渡す
        pass
    
    def run_loop(self):
        """自動トレーニングループを実行"""
        print(f"Starting auto-training loop ({self.num_loops} iterations)")
        print(f"Log directory: {self.log_dir}")
        
        for loop_num in range(1, self.num_loops + 1):
            print(f"\n\n{'#'*60}")
            print(f"# LOOP {loop_num}/{self.num_loops}")
            print(f"{'#'*60}")
            
            # 1. トレーニング実行
            success, exp_name = self.run_training(loop_num)
            if not success:
                print(f"Training failed at loop {loop_num}")
                break
            
            # 2. 評価実行
            score = self.run_evaluation(exp_name)
            self.reward_history.append({
                "loop": loop_num,
                "exp_name": exp_name,
                "score": score,
                "rewards": self.reward_scales.copy()
            })
            
            # ベストスコアを更新
            if score > self.best_score:
                self.best_score = score
                self.best_reward_config = self.reward_scales.copy()
                print(f"✓ New best score: {self.best_score:.4f}")
            
            # 3. リワード設定を保存
            self.save_reward_config(loop_num)
            
            # 4. リワードを調整（最後のループでない場合）
            if loop_num < self.num_loops:
                self.adjust_rewards(loop_num, score)
        
        # 最終結果を表示
        self.print_final_results()
    
    def print_final_results(self):
        """最終結果をサマリー表示"""
        print(f"\n\n{'='*60}")
        print("AUTO-TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"\nBest reward configuration:")
        if self.best_reward_config:
            for key, value in self.best_reward_config.items():
                print(f"  {key}: {value:.4f}")
        
        print(f"\nTraining history:")
        for entry in self.reward_history:
            print(f"  Loop {entry['loop']}: {entry['exp_name']}, Score: {entry['score']:.4f}")
        
        # 結果を JSON に保存
        results_file = os.path.join(self.log_dir, "final_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "best_score": self.best_score,
                "best_config": self.best_reward_config,
                "history": self.reward_history
            }, f, indent=2)
        print(f"\nResults saved to {results_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="hrp2-walking")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--num_loops", type=int, default=5)
    parser.add_argument("--force_forward", action="store_true", 
                       help="Force forward velocity optimization (for stationary model fix)")
    args = parser.parse_args()
    
    optimizer = RewardOptimizer(
        base_exp_name=args.exp_name,
        max_iterations=args.max_iterations,
        num_loops=args.num_loops
    )
    
    # 前進強制モード
    if args.force_forward:
        print("\n" + "="*60)
        print("⚠️  FORCE FORWARD MODE - Prioritizing velocity")
        print("="*60)
        optimizer.reward_scales["tracking_lin_vel"] = 10.0  # 最大値
        optimizer.reward_scales["feet_air_time"] = 0.5
        optimizer.reward_scales["feet_alternating_pos"] = 0.5
        optimizer.print_reward_config()
    
    optimizer.run_loop()

if __name__ == "__main__":
    main()