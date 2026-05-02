"""
On-policy GRPO trainer for ALFWorld with optional selective-rollout gate.

Architecture (single Python process, two GPUs per run):
  - GPU A:  VLLM (Qwen2.5-7B + enable_lora=True) for rollout
  - GPU B:  HF transformers + peft LoRA for forward+backward of policy-grad loss

Per training iteration:
  1. VLLM rolls out N prompts × G trajectories with the *current* LoRA path
     (or no adapter on iter 0).  If --gate is on, trajectories of zero-variance-
     predicted groups are halted at step K (the rollout already supports this).
  2. Compute GRPO advantages from the terminal rewards.
  3. HF computes log-prob of the recorded action tokens, weighted PG loss,
     backward, optimizer.step().
  4. Save the updated LoRA to a new directory; point VLLM at it.
  5. Every --eval-every iterations, run a held-out eval pass on 50 valid_seen
     tasks (uses the same VLLM with the latest LoRA, no training).

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/19_grpo_onpolicy.py \
        --gate off --iters 60 --prompts-per-iter 10 --group-size 8 \
        --eval-every 10 --eval-tasks 50 --seed 42 --out runs/onpolicy_baseline
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.divergence import divergence_at_K  # noqa: E402
from src.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    followup_user_message,
    parse_action,
)
from src.env import GroupEnv, list_games  # noqa: E402


# ---------------------------------------------------------------------------
# Rollout (replicates src/rollout.py:rollout_group but with VLLM directly so
# we don't import src.llm.ChatLLM until after CUDA_VISIBLE_DEVICES is set).
# ---------------------------------------------------------------------------
def rollout_one_group(llm, game_file, task_id, group_size, max_steps,
                      temperature, max_new_tokens, gate=None):
    from src.rollout import rollout_group
    return rollout_group(
        llm=llm, game_file=game_file, task_id=task_id,
        group_size=group_size, max_steps=max_steps,
        temperature=temperature, max_new_tokens=max_new_tokens,
        gate=gate,
    )


# ---------------------------------------------------------------------------
# Training: build (chat_text, assistant_text, advantage) items from rollouts.
# ---------------------------------------------------------------------------
def build_train_items(rollouts: List[Any], gate_active: bool, K: int,
                      d_low: float, t_high: float):
    """Yield (chat_history_messages, assistant_text, advantage_for_this_traj,
    is_cut, group_idx, traj_idx, step_idx)."""
    items = []
    n_groups_cut = 0
    n_groups_zero_var = 0
    for gi, rg in enumerate(rollouts):
        rewards = np.array([t.total_reward for t in rg.trajectories])
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        is_cut = False
        if gate_active and len(rg.trajectories[0].steps) >= K:
            actions_per_traj = [
                [s.action for s in t.steps[:K]] for t in rg.trajectories
            ]
            obs_per_traj = [
                [s.obs for s in t.steps[:K]] for t in rg.trajectories
            ]
            metrics = divergence_at_K(actions_per_traj, obs_per_traj, K=K)
            d_K = metrics["prefix_edit_distance_mean"]
            t_K = metrics["termination_fraction"]
            is_cut = (d_K < d_low) or (t_K >= t_high)
        if is_cut:
            n_groups_cut += 1
        if rewards.std() == 0:
            n_groups_zero_var += 1
        # If cut, skip — no contribution to loss.
        if is_cut:
            continue
        for ti, traj in enumerate(rg.trajectories):
            for si, step in enumerate(traj.steps):
                items.append({
                    "group_idx": gi, "traj_idx": ti, "step_idx": si,
                    "advantage": float(adv[ti]),
                    "step": step,
                    "trajectory": traj,
                    "rg": rg,
                })
    return items, n_groups_cut, n_groups_zero_var


# ---------------------------------------------------------------------------
# Reconstruct chat at (traj_idx, step_idx) and tokenize for HF.
# ---------------------------------------------------------------------------
def chat_for_step(rg, ti: int, si: int):
    steps = rg.trajectories[ti].steps
    chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    if si >= len(steps):
        return None
    chat.append({"role": "user",
                 "content": f"Observation: {steps[0].obs}"})
    for k in range(si):
        s = steps[k]
        chat.append({
            "role": "assistant",
            "content": f"Thought: {s.thought}\nAction: {s.action}",
        })
        if k + 1 < len(steps):
            chat.append({
                "role": "user",
                "content": f"Observation: {steps[k + 1].obs}",
            })
    s = steps[si]
    asst = f"Thought: {s.thought}\nAction: {s.action}"
    return chat, asst


def hf_pg_step(model, tokenizer, items, n_chunks_for_logits=8,
               max_train_items=128, rng=None):
    """One PG step: forward on each item, backward into LoRA, optimizer.step
    happens outside.  Returns (loss_value, count, sum_grad_norm_proxy)."""
    if rng is None:
        rng = random.Random(0)
    device = next(model.parameters()).device
    # Subsample if too many items to keep iteration time bounded.
    if len(items) > max_train_items:
        items = rng.sample(items, max_train_items)
    n = len(items)
    if n == 0:
        return 0.0, 0
    total_loss_val = 0.0
    for it in items:
        rg = it["rg"]; ti = it["traj_idx"]; si = it["step_idx"]
        adv = it["advantage"]
        built = chat_for_step(rg, ti, si)
        if built is None:
            continue
        chat, asst = built
        chat_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
        full = chat_text + asst
        prefix_ids = tokenizer(chat_text, add_special_tokens=False)["input_ids"]
        enc = tokenizer(full, return_tensors="pt", padding=False,
                        truncation=True, max_length=1024).to(device)
        if enc["input_ids"].size(1) <= len(prefix_ids):
            continue
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"])
        logits = out.logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:].clone()
        attn = enc["attention_mask"][:, 1:]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        start = max(0, len(prefix_ids) - 1)
        if start < labels.size(1):
            mask[:, start:] = (attn[:, start:] == 1)
        if mask.sum() == 0:
            continue
        T = logits.size(1)
        chunk = max(1, T // n_chunks_for_logits)
        logp_total = torch.zeros((), device=device, dtype=torch.float32)
        cnt = 0
        for s_idx in range(0, T, chunk):
            e_idx = min(s_idx + chunk, T)
            sub_logits = logits[:, s_idx:e_idx, :]
            sub_labels = labels[:, s_idx:e_idx]
            sub_mask = mask[:, s_idx:e_idx]
            nll = F.cross_entropy(
                sub_logits.reshape(-1, sub_logits.size(-1)),
                sub_labels.reshape(-1), reduction="none",
            ).view_as(sub_labels).float()
            logp_total = logp_total - (nll * sub_mask).sum()
            cnt += int(sub_mask.sum().item())
            del sub_logits, nll
        if cnt == 0:
            continue
        mean_logp = logp_total / cnt
        loss_b = -(adv * mean_logp) / n
        loss_b.backward()
        total_loss_val += float(loss_b.detach()) * n
        del out, logits, labels, mask, mean_logp, loss_b, enc
        torch.cuda.empty_cache()
    return total_loss_val / n, n


# ---------------------------------------------------------------------------
# Eval: run held-out tasks with current model, return success rate.
# ---------------------------------------------------------------------------
def run_eval(llm, eval_games, group_size_for_eval, max_steps, temperature,
             max_new_tokens):
    """Run each held-out game once (G=1 effectively, single trajectory) and
    measure success rate."""
    successes = 0; total = 0
    for gf in eval_games:
        tid = "/".join(gf.split("/")[-3:-1])
        try:
            rg = rollout_one_group(
                llm, gf, tid, group_size=group_size_for_eval,
                max_steps=max_steps, temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f"  eval err {tid}: {type(e).__name__}: {e}", flush=True)
            continue
        # success = any trajectory won
        any_win = any(t.won for t in rg.trajectories)
        if any_win:
            successes += 1
        total += 1
    return successes, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gate", choices=["on", "off"], default="off")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--d-low", type=float, default=0.12)
    ap.add_argument("--t-high", type=float, default=0.90)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--prompts-per-iter", type=int, default=10)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-rollout-steps", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-train-items-per-iter", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-tasks", type=int, default=50)
    ap.add_argument("--eval-max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vllm-gpu", type=int, default=0,
                    help="device index *within CUDA_VISIBLE_DEVICES* for VLLM")
    ap.add_argument("--hf-gpu", type=int, default=1,
                    help="device index *within CUDA_VISIBLE_DEVICES* for HF training")
    ap.add_argument("--out", default="runs/onpolicy")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = out / "train.jsonl"
    eval_log = out / "eval.jsonl"
    lora_dir = out / "lora_adapters"
    lora_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # ---- split games into train pool / held-out eval set ----
    games = list_games("valid_seen")
    rng.shuffle(games)
    eval_games = games[: args.eval_tasks]
    train_games = games[args.eval_tasks:]
    print(f"[trainer] {len(train_games)} train games, {len(eval_games)} eval games",
          flush=True)

    # ---- VLLM (rollout side, gpu A) ----
    # Pin VLLM to its GPU index by setting CUDA_VISIBLE_DEVICES BEFORE the
    # vllm import.  We do this by spawning the rollout in a subprocess?  No —
    # since we have CUDA_VISIBLE_DEVICES already set externally to the two
    # GPUs we want, we just rely on torch.cuda.set_device for HF.  VLLM picks
    # cuda:0 (the first visible device).
    print(f"[trainer] loading VLLM on visible-device-{args.vllm_gpu} ...",
          flush=True)
    # Workaround: we want VLLM on visible_dev[vllm_gpu] and HF on
    # visible_dev[hf_gpu].  Without process isolation we can only put VLLM on
    # CUDA:0 (its default).  For now require: CUDA_VISIBLE_DEVICES sees [A, B]
    # and we rely on VLLM=A, HF=B by setting torch.cuda.set_device(B) for HF.
    from src.llm import ChatLLM
    llm = ChatLLM(
        model=args.model,
        gpu_memory_utilization=args.vllm_gpu_mem,
        max_model_len=16384,
        enable_lora=True,
        max_lora_rank=args.lora_rank,
    )
    print("[trainer] VLLM ready", flush=True)

    # ---- HF training side (gpu B) ----
    print(f"[trainer] loading HF model on visible-device-{args.hf_gpu} ...",
          flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
        device_map={"": args.hf_gpu},
    )
    base.config.use_cache = False
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    print("[trainer] HF ready", flush=True)

    gate_dict = ({"K": args.K, "d_low": args.d_low, "t_high": args.t_high}
                 if args.gate == "on" else None)
    train_iter = iter(rng.sample(train_games, len(train_games)))

    def next_prompt():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(rng.sample(train_games, len(train_games)))
            return next(train_iter)

    # ---- Pre-eval ----
    print("[eval] iter 0 (base model) ...", flush=True)
    t_e = time.time()
    succ, tot = run_eval(llm, eval_games, group_size_for_eval=1,
                          max_steps=args.max_rollout_steps,
                          temperature=0.0,  # greedy for eval
                          max_new_tokens=args.eval_max_new_tokens)
    eval_rec = {"iter": 0, "success_rate": succ / max(tot, 1),
                "n_success": succ, "n_total": tot,
                "wall_time_sec": time.time() - t_e}
    with open(eval_log, "a") as f:
        f.write(json.dumps(eval_rec) + "\n")
    print(f"[eval] iter 0: {succ}/{tot} = {succ/max(tot,1)*100:.1f}%  "
          f"({eval_rec['wall_time_sec']:.0f}s)", flush=True)

    # ---- Training loop ----
    cumulative_wall_clock = 0.0
    for it in range(1, args.iters + 1):
        t_iter = time.time()
        # Rollout
        t_r = time.time()
        prompts = [next_prompt() for _ in range(args.prompts_per_iter)]
        rollouts = []
        for gf in prompts:
            tid = "/".join(gf.split("/")[-3:-1])
            try:
                rg = rollout_one_group(
                    llm, gf, tid, group_size=args.group_size,
                    max_steps=args.max_rollout_steps,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    gate=gate_dict,
                )
                rollouts.append(rg)
            except Exception as e:
                print(f"  rollout err {tid}: {type(e).__name__}: {e}", flush=True)
        rollout_secs = time.time() - t_r

        # Build train items
        items, n_cut, n_zv = build_train_items(
            rollouts, gate_active=(args.gate == "on"),
            K=args.K, d_low=args.d_low, t_high=args.t_high,
        )

        # Train one step
        t_t = time.time()
        if items:
            optimizer.zero_grad()
            loss_val, n_used = hf_pg_step(
                model, tok, items,
                max_train_items=args.max_train_items_per_iter, rng=rng,
            )
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0))
            optimizer.step()
        else:
            loss_val = 0.0; n_used = 0; grad_norm = 0.0
        train_secs = time.time() - t_t

        # Save updated LoRA + hot-swap into VLLM
        t_s = time.time()
        new_lora_path = lora_dir / f"iter_{it:04d}"
        if new_lora_path.exists():
            shutil.rmtree(new_lora_path)
        new_lora_path.mkdir()
        model.save_pretrained(str(new_lora_path))
        # Keep at most last 3 adapters to save disk
        for old in sorted(lora_dir.iterdir()):
            keep = sorted(lora_dir.iterdir())[-3:]
            if old not in keep:
                shutil.rmtree(old, ignore_errors=True)
        llm.set_lora(str(new_lora_path))
        save_secs = time.time() - t_s

        rewards_mean = float(np.mean([
            np.mean([t.total_reward for t in rg.trajectories]) for rg in rollouts
        ])) if rollouts else 0.0
        rewards_max = float(np.max([
            np.max([t.total_reward for t in rg.trajectories]) for rg in rollouts
        ])) if rollouts else 0.0

        iter_secs = time.time() - t_iter
        cumulative_wall_clock += iter_secs

        rec = {
            "iter": it, "gate": args.gate,
            "n_groups": len(rollouts),
            "n_groups_cut": n_cut,
            "n_zero_variance": n_zv,
            "n_train_items": n_used,
            "loss": loss_val, "grad_norm": grad_norm,
            "rewards_mean": rewards_mean, "rewards_max": rewards_max,
            "rollout_secs": rollout_secs,
            "train_secs": train_secs,
            "save_secs": save_secs,
            "iter_secs": iter_secs,
            "cumulative_wall_clock_sec": cumulative_wall_clock,
        }
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[iter {it:03d}] gate={args.gate} cut={n_cut}/{len(rollouts)} "
              f"zv={n_zv}/{len(rollouts)} items={n_used} "
              f"loss={loss_val:+.4f} gn={grad_norm:.3f} R̄={rewards_mean:+.3f} "
              f"rollout={rollout_secs:.0f}s train={train_secs:.0f}s "
              f"save={save_secs:.0f}s iter={iter_secs:.0f}s "
              f"cum={cumulative_wall_clock:.0f}s",
              flush=True)

        # Periodic eval
        if it % args.eval_every == 0 or it == args.iters:
            t_e = time.time()
            succ, tot = run_eval(
                llm, eval_games, group_size_for_eval=1,
                max_steps=args.max_rollout_steps,
                temperature=0.0, max_new_tokens=args.eval_max_new_tokens,
            )
            eval_rec = {
                "iter": it, "success_rate": succ / max(tot, 1),
                "n_success": succ, "n_total": tot,
                "wall_time_sec": time.time() - t_e,
                "cumulative_wall_clock_sec": cumulative_wall_clock,
            }
            with open(eval_log, "a") as f:
                f.write(json.dumps(eval_rec) + "\n")
            print(f"[eval] iter {it}: {succ}/{tot} = {succ/max(tot,1)*100:.1f}%  "
                  f"({eval_rec['wall_time_sec']:.0f}s)  "
                  f"cum={cumulative_wall_clock:.0f}s", flush=True)

    print(f"[trainer] done. {args.iters} iters in {cumulative_wall_clock:.0f}s.",
          flush=True)


if __name__ == "__main__":
    main()
