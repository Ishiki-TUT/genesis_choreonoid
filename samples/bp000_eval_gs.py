import argparse
import os
import pickle

from importlib import metadata
ver_rsl_rl = metadata.version("rsl-rl-lib")
print(f"Version of rsl-rl-lib : {ver_rsl_rl}")

import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

from bp000_env_gs import BP000EnvGenesis as RLEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="bp000-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--action_scale", type=float, default=1.0)
    args = parser.parse_args()

    ## set robot path fix collisiton 
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # /userdir
    robot_path = os.path.join(ROOT, "userdir", "humanoid_research_k", "robots", "kawada_base.simple_collision.urdf")

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    global action_scale
    action_scale = args.action_scale

    ## override
    env_cfg["episode_length_s"] = 40.0
    command_cfg["lin_vel_x_range"] = [0.5, 0.5]
    env_cfg['base_roll_noise'] = [0,0]
    env_cfg['base_pitch_noise'] = [0,0]

    env = RLEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        dt=env_cfg['dt'],
        substeps=env_cfg['substeps'],
        show_viewer=True,
        robot_urdf_path=robot_path,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device='cuda')
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device='cuda')

    return env, policy

def eval_policy(env, policy):
    obs, _ = env.reset()
    
    cnt = 0

    print("env_reset:", obs["policy"])
    with torch.no_grad():
        while True:
            actions = policy(obs)
            scaled_actions = actions * action_scale
            obs, rews, dones, infos = env.step(scaled_actions)
            print(obs["policy"])
            print(f"Step: {cnt}")
            input()

            cnt += 1
if __name__ == "__main__":
    env, policy = main()
    eval_policy(env, policy)

"""
# evaluation
python examples/locomotion/bp000_eval.py -e bp000-walking --ckpt 100
"""