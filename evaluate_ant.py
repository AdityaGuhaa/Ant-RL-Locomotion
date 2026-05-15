"""Evaluate a trained PPO Ant-Stand agent.

Combines the previous evaluate_ant.py (visual + console prints) and
evaluate_agent.py (CSV logging) into one script.

Examples:
    # Watch the ant with console prints (old evaluate_ant.py behavior)
    python evaluate_ant.py

    # Log per-step metrics to eval_performance_log.csv (old evaluate_agent.py behavior)
    python evaluate_ant.py --csv --no-print --no-sleep

    # Both: watch slowly, print, and log to CSV
    python evaluate_ant.py --csv

    # Headless eval, no GUI window
    python evaluate_ant.py --no-render --csv --no-sleep
"""

import argparse
import csv
import os
import time

from stable_baselines3 import PPO

from handstand_rl.envs.ant_stand_env import AntStandEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Ant-Stand agent.")
    parser.add_argument("--model", default="ppo_ant_stand", help="Path to the saved PPO model (without .zip).")
    parser.add_argument("--steps", type=int, default=1000, help="Number of evaluation steps.")
    parser.add_argument("--csv-path", default="eval_performance_log.csv", help="Where to write the CSV log.")
    parser.add_argument("--csv", action="store_true", help="Write per-step metrics to CSV.")
    parser.add_argument("--no-print", action="store_true", help="Disable per-step console prints.")
    parser.add_argument("--no-sleep", action="store_true", help="Disable the 0.02s per-step sleep.")
    parser.add_argument("--sleep", type=float, default=0.02, help="Seconds to sleep between steps when sleeping is on.")
    parser.add_argument("--no-render", action="store_true", help="Run headless (no MuJoCo viewer window).")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using deterministic=True.")
    return parser.parse_args()


def main():
    args = parse_args()

    render_mode = None if args.no_render else "human"
    deterministic = not args.stochastic

    model = PPO.load(args.model)
    env = AntStandEnv(render_mode=render_mode)
    obs, _ = env.reset()

    log_file = None
    log_writer = None
    if args.csv:
        log_path = os.path.abspath(args.csv_path)
        log_file = open(log_path, mode="w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Step", "Reward", "TorsoHeight", "Terminated", "Truncated"])

    try:
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            torso_height = float(obs[0])

            if log_writer is not None:
                log_writer.writerow([step, float(reward), torso_height, bool(terminated), bool(truncated)])

            if not args.no_print:
                print(
                    f"[STEP {step}] Reward: {float(reward):.2f} | "
                    f"TorsoHeight: {torso_height:.3f} | "
                    f"Terminated: {bool(terminated)} | Truncated: {bool(truncated)}"
                )

            if not args.no_sleep:
                time.sleep(args.sleep)

            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        if log_file is not None:
            log_file.close()
        env.close()


if __name__ == "__main__":
    main()
