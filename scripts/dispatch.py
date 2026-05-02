"""
Multi-GPU supervisor + worker for selective-rollout data collection.

Usage:
    # init a shared queue of 100 tasks, then supervise GPUs dynamically
    python scripts/dispatch.py supervise \
        --num-tasks 100 --group-size 8 --max-steps 30 --seed 42 \
        --queue data/queue.txt --out data/rollouts.jsonl \
        --poll-interval 30 --free-mem-mib 28000

    # (called by supervisor per GPU; do not run manually)
    python scripts/dispatch.py worker --gpu 0 \
        --queue data/queue.txt --out data/rollouts.jsonl \
        --group-size 8 --max-steps 30

The supervisor:
  1. Seeds a shared queue file with N game paths (one per line).
  2. Spawns a worker on every GPU that currently satisfies
     memory.free >= --free-mem-mib AND no worker is already running on it.
  3. Polls nvidia-smi every --poll-interval seconds; spawns new workers as
     additional GPUs free up.
  4. Exits when the queue is empty AND every spawned worker has finished.

Each worker loads Qwen2.5-7B once, then pops tasks one at a time, appending
each finished group's JSON to --out (lock-protected).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

from src import queue_fs  # noqa: E402


# ---------- GPU polling ----------
def query_gpus() -> List[Dict]:
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for ln in out.strip().splitlines():
        i, mf, mu, u = [s.strip() for s in ln.split(",")]
        rows.append({"index": int(i), "mem_free": int(mf),
                     "mem_used": int(mu), "util": int(u)})
    return rows


# ---------- worker subcommand ----------
def run_worker(args: argparse.Namespace):
    # Isolate this process to the chosen GPU; VLLM will see it as device 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Late import so CUDA_VISIBLE_DEVICES is respected
    from src.llm import ChatLLM
    from src.rollout import rollout_group

    log_prefix = f"[worker gpu={args.gpu} pid={os.getpid()}]"
    print(f"{log_prefix} loading model {args.model} …", flush=True)

    llm = ChatLLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enable_lora=bool(getattr(args, "lora_path", "")),
    )
    if getattr(args, "lora_path", ""):
        llm.set_lora(args.lora_path)
        print(f"{log_prefix} loaded LoRA from {args.lora_path}", flush=True)

    print(f"{log_prefix} model ready. polling queue {args.queue}", flush=True)

    done = 0
    t0 = time.time()
    while True:
        item = queue_fs.pop(args.queue)
        if item is None:
            print(f"{log_prefix} queue empty. processed {done} tasks in "
                  f"{time.time() - t0:.0f}s", flush=True)
            return

        task_id = "/".join(item.split("/")[-3:-1])
        print(f"{log_prefix} START {task_id}", flush=True)
        t_task = time.time()
        gate_kwargs = None
        if getattr(args, "gate_K", 0) > 0:
            gate_kwargs = {"K": args.gate_K, "d_low": args.gate_d_low,
                           "t_high": args.gate_t_high}
        try:
            rg = rollout_group(
                llm=llm,
                game_file=item,
                task_id=task_id,
                group_size=args.group_size,
                max_steps=args.max_steps,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                gate=gate_kwargs,
            )
        except Exception as e:
            print(f"{log_prefix} ERROR {task_id}: {type(e).__name__}: {e}",
                  flush=True)
            queue_fs.append_line(args.failed_log, f"{item}\t{type(e).__name__}: {e}")
            continue

        wins = sum(1 for t in rg.trajectories if t.won)
        rewards = [t.total_reward for t in rg.trajectories]
        queue_fs.append_line(args.out, json.dumps(rg.to_dict()))
        done += 1
        print(f"{log_prefix} DONE  {task_id}  wins={wins}/{rg.group_size}  "
              f"rewards={rewards}  wall={time.time() - t_task:.1f}s  "
              f"total_done={done}", flush=True)


# ---------- supervisor subcommand ----------
def _spawn_worker(args, gpu: int, log_path: str) -> subprocess.Popen:
    cmd = [
        sys.executable, str(THIS),
        "worker",
        "--gpu", str(gpu),
        "--queue", args.queue,
        "--out", args.out,
        "--failed-log", args.failed_log,
        "--group-size", str(args.group_size),
        "--max-steps", str(args.max_steps),
        "--temperature", str(args.temperature),
        "--max-new-tokens", str(args.max_new_tokens),
        "--gpu-mem-util", str(args.gpu_mem_util),
        "--max-model-len", str(args.max_model_len),
        "--model", args.model,
        "--gate-K", str(getattr(args, "gate_K", 0)),
        "--gate-d-low", str(getattr(args, "gate_d_low", 0.12)),
        "--gate-t-high", str(getattr(args, "gate_t_high", 0.90)),
        "--lora-path", str(getattr(args, "lora_path", "") or ""),
    ]
    log_file = open(log_path, "a")
    log_file.write(f"\n\n===== worker start gpu={gpu} ts={time.strftime('%F %T')} cmd={' '.join(shlex.quote(c) for c in cmd)} =====\n")
    log_file.flush()
    p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return p


def run_supervise(args: argparse.Namespace):
    from src.env import list_games

    # --- init queue ---
    rng = random.Random(args.seed)
    games = list_games(args.split)
    assert len(games) >= args.num_tasks, f"{len(games)} games < {args.num_tasks}"
    rng.shuffle(games)
    games = games[: args.num_tasks]

    # Reset / init the queue and output files
    Path(args.queue).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.resume and Path(args.out).exists():
        # skip games that already have a record in args.out
        done_ids = set()
        with open(args.out) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    done_ids.add(r["game_file"])
                except Exception:
                    pass
        games = [g for g in games if g not in done_ids]
        print(f"[supervise] resume mode: {len(done_ids)} already done, "
              f"{len(games)} remaining")
    else:
        open(args.out, "w").close()
        open(args.failed_log, "w").close()

    queue_fs.init(args.queue, games)
    total_tasks = len(games)
    print(f"[supervise] queue seeded with {total_tasks} tasks")

    workers_dir = Path(args.out).parent / "worker_logs"
    workers_dir.mkdir(exist_ok=True)

    # gpu_index -> subprocess.Popen
    workers: Dict[int, subprocess.Popen] = {}

    def _cleanup(*_):
        print("\n[supervise] interrupt — terminating workers …", flush=True)
        for g, p in workers.items():
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(2)
        for g, p in workers.items():
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        sys.exit(130)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    t0 = time.time()
    last_status = 0.0
    while True:
        # Reap finished workers
        alive: Dict[int, subprocess.Popen] = {}
        for gpu, p in workers.items():
            rc = p.poll()
            if rc is None:
                alive[gpu] = p
            else:
                print(f"[supervise] worker gpu={gpu} exited rc={rc}")
        workers = alive

        remaining = queue_fs.remaining(args.queue)

        # Termination?
        if remaining == 0 and not workers:
            # Give any still-running worker a moment (it might be wrapping up a task)
            break

        # Try to spawn a new worker on free GPU
        if remaining > 0:
            gpus = query_gpus()
            # sort by free memory desc, skip GPUs already in workers
            candidates = [g for g in gpus if g["index"] not in workers
                          and g["mem_free"] >= args.free_mem_mib]
            # also require util <= threshold to avoid grabbing a GPU someone is
            # actively transferring data to
            candidates = [g for g in candidates if g["util"] <= args.max_util_pct]
            candidates.sort(key=lambda g: -g["mem_free"])

            for g in candidates[: max(0, args.max_workers - len(workers))]:
                gi = g["index"]
                print(f"[supervise] free GPU detected: index={gi} "
                      f"mem_free={g['mem_free']}MiB util={g['util']}% "
                      f"— spawning worker", flush=True)
                log_path = str(workers_dir / f"gpu{gi}.log")
                workers[gi] = _spawn_worker(args, gi, log_path)
                time.sleep(2)  # small stagger so workers don't race on first pop

        now = time.time()
        if now - last_status >= args.status_interval:
            elapsed = now - t0
            completed = total_tasks - remaining - sum(1 for p in workers.values() if p.poll() is None and False)  # approx
            # actual completed = lines in args.out
            try:
                with open(args.out) as f:
                    completed = sum(1 for _ in f)
            except FileNotFoundError:
                completed = 0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (remaining + len(workers)) / rate if rate > 0 else float("inf")
            gpus_used = ",".join(str(g) for g in sorted(workers))
            print(f"[supervise] t+{elapsed:.0f}s  done {completed}/{total_tasks}  "
                  f"queue_rem={remaining}  workers=[{gpus_used}]  "
                  f"rate={rate:.2f}/s  eta={eta:.0f}s", flush=True)
            last_status = now

        time.sleep(args.poll_interval)

    elapsed = time.time() - t0
    print(f"\n[supervise] ALL DONE.  {total_tasks} tasks in {elapsed:.0f}s  "
          f"({total_tasks / elapsed:.2f} tasks/sec)")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- common flags applied to both subcommands ----
    def _common(p):
        p.add_argument("--queue", default="data/queue.txt")
        p.add_argument("--out", default="data/rollouts.jsonl")
        p.add_argument("--failed-log", default="data/failed.log")
        p.add_argument("--group-size", type=int, default=8)
        p.add_argument("--max-steps", type=int, default=30)
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--max-new-tokens", type=int, default=96)
        p.add_argument("--gpu-mem-util", type=float, default=0.85)
        p.add_argument("--max-model-len", type=int, default=16384)
        p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
        p.add_argument("--gate-K", type=int, default=0,
                       help="Selective-rollout gate trigger step (0 = no gate)")
        p.add_argument("--gate-d-low", type=float, default=0.12)
        p.add_argument("--gate-t-high", type=float, default=0.90)
        p.add_argument("--lora-path", default="",
                       help="Optional LoRA adapter directory; empty = base model")

    sup = sub.add_parser("supervise")
    _common(sup)
    sup.add_argument("--num-tasks", type=int, default=100)
    sup.add_argument("--seed", type=int, default=42)
    sup.add_argument("--split", default="valid_seen")
    sup.add_argument("--poll-interval", type=float, default=30,
                     help="seconds between GPU polls (and worker reaping)")
    sup.add_argument("--status-interval", type=float, default=60,
                     help="seconds between human-readable status lines")
    sup.add_argument("--free-mem-mib", type=int, default=28000,
                     help="min GPU free memory to consider a GPU usable")
    sup.add_argument("--max-util-pct", type=int, default=15,
                     help="max GPU util %% to consider a GPU idle")
    sup.add_argument("--max-workers", type=int, default=4,
                     help="cap on simultaneous workers (each worker ≈ 1 GPU)")
    sup.add_argument("--resume", action="store_true",
                     help="skip tasks already present in --out")

    wk = sub.add_parser("worker")
    _common(wk)
    wk.add_argument("--gpu", type=int, required=True)

    args = ap.parse_args()
    if args.cmd == "worker":
        run_worker(args)
    else:
        run_supervise(args)


if __name__ == "__main__":
    main()
