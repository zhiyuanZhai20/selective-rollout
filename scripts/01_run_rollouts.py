"""
Run G=8 parallel rollouts on N ALFWorld tasks, save to JSONL.

Usage:
    python scripts/01_run_rollouts.py --num-tasks 100 --group-size 8 \
        --out data/rollouts.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from src.env import list_games  # noqa: E402
from src.llm import ChatLLM      # noqa: E402
from src.rollout import rollout_group  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid_seen")
    ap.add_argument("--num-tasks", type=int, default=100)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    games = list_games(args.split)
    print(f"[rollouts] found {len(games)} games in split={args.split}")
    assert len(games) >= args.num_tasks, f"not enough games ({len(games)}) for {args.num_tasks} tasks"

    rng.shuffle(games)
    games = games[: args.num_tasks]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Truncate output file
    open(args.out, "w").close()

    llm = ChatLLM(model=args.model, gpu_memory_utilization=args.gpu_mem_util)

    t_start = time.time()
    for idx, g in enumerate(games):
        task_id = "/".join(g.split("/")[-3:-1])
        print(f"\n[{idx + 1}/{args.num_tasks}] {task_id}")
        try:
            rg = rollout_group(
                llm=llm,
                game_file=g,
                task_id=task_id,
                group_size=args.group_size,
                max_steps=args.max_steps,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"  ! error: {type(e).__name__}: {e}")
            continue
        rewards = [t.total_reward for t in rg.trajectories]
        wins = sum(1 for t in rg.trajectories if t.won)
        steps = [len(t.steps) for t in rg.trajectories]
        print(f"  wins {wins}/{rg.group_size}  reward {rewards}  steps {steps}  "
              f"wall {rg.wall_time_sec:.1f}s")
        with open(args.out, "a") as f:
            f.write(json.dumps(rg.to_dict()) + "\n")

        elapsed = time.time() - t_start
        rate = (idx + 1) / elapsed
        eta = (args.num_tasks - idx - 1) / rate if rate > 0 else 0
        print(f"  cumulative: {elapsed:.0f}s, ETA {eta:.0f}s")

    print(f"\n[rollouts] done — wrote to {args.out}")


if __name__ == "__main__":
    main()
