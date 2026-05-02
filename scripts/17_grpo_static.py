"""
Phase F.3 — off-policy GRPO trainer over the existing 100-task rollout buffer.

Why this exists:
    Real on-policy GRPO with HF generate is bottlenecked by env rollout time
    (3-6 min per training step on 7B + 4 GPU). Across 12 steps that's 40-80 min,
    most of it producing zero-variance groups that contribute no gradient.

    This script trades on-policy fidelity for speed by replaying the
    100-task rollout buffer collected for the wall-clock A/B
    (data/rollouts.jsonl). Each "training step" pulls a batch of groups,
    applies the gate (or not), computes GRPO advantages from the stored
    rewards, and takes one optimizer step on the LoRA adapter using the
    log-probs of the recorded actions. No env stepping, no LLM generation
    during training.

    This is exactly the "REINFORCE on a fixed rollout buffer" recipe used
    by many off-policy LLM-RL baselines (DPO, KTO, etc.) — and it lets us
    cleanly demonstrate
        (i)  the trainer's gradient updates work,
        (ii) the gate's compute trade-off shows up in per-step wall-clock,
        (iii) the loss/grad-norm trajectory is similar with vs without the
             gate (since cut groups contribute zero gradient anyway).

Usage:
    python scripts/17_grpo_static.py --gate off --out runs/static_baseline
    python scripts/17_grpo_static.py --gate on  --out runs/static_gated
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.divergence import divergence_at_K  # noqa: E402
from src.prompts import SYSTEM_PROMPT, followup_user_message  # noqa: E402


def load_groups(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def gate_fires(group: Dict[str, Any], K: int, d_low: float, t_high: float) -> bool:
    """Replay the gate decision on a stored group's first K steps."""
    actions_per_traj = [
        [s["action"] for s in t["steps"][:K]]
        for t in group["trajectories"]
    ]
    obs_per_traj = [
        [s["obs"] for s in t["steps"][:K]]
        for t in group["trajectories"]
    ]
    G = len(actions_per_traj)
    if G < 2:
        return False
    if any(len(a) < 1 for a in actions_per_traj):
        return False
    metrics = divergence_at_K(actions_per_traj, obs_per_traj, K=K)
    d_K = metrics["prefix_edit_distance_mean"]
    t_K = metrics["termination_fraction"]
    return (d_K < d_low) or (t_K >= t_high)


def grpo_advantages(rewards: List[float]) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    mu = r.mean()
    sigma = r.std()
    return (r - mu) / (sigma + 1e-8)


def build_chat_for_step(group: Dict[str, Any], traj_idx: int, step_idx: int):
    """Reconstruct the chat history that the model saw at (traj_idx, step_idx)
    and return (chat_history_messages, assistant_text).
    """
    steps = group["trajectories"][traj_idx]["steps"]
    chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Initial user message — we only have the initial obs from steps[0]
    if step_idx >= len(steps):
        return None
    first = steps[0]
    chat.append({"role": "user", "content": f"Observation: {first['obs']}"})
    for i in range(step_idx):
        s = steps[i]
        thought = s.get("thought", "")
        action = s["action"]
        chat.append({
            "role": "assistant",
            "content": f"Thought: {thought}\nAction: {action}"
        })
        if i + 1 < len(steps):
            nxt = steps[i + 1]
            chat.append({
                "role": "user",
                "content": f"Observation: {nxt['obs']}"
            })
    s = steps[step_idx]
    assistant_text = f"Thought: {s.get('thought', '')}\nAction: {s['action']}"
    return chat, assistant_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--rollouts", default="data/rollouts.jsonl")
    ap.add_argument("--gate", choices=["on", "off"], default="off")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--d_low", type=float, default=0.12)
    ap.add_argument("--t_high", type=float, default=0.90)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--groups-per-step", type=int, default=4)
    ap.add_argument("--max-train-steps-per-traj", type=int, default=8)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/static_grpo")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.jsonl"

    rng = random.Random(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Load buffer
    groups = load_groups(ROOT / args.rollouts)
    print(f"[trainer] loaded {len(groups)} groups from {args.rollouts}", flush=True)

    # Pre-compute gate decisions on all groups (deterministic over fixed buffer).
    # When --gate off, this still gives us a reference statistic but we DO NOT
    # skip cut groups in the loss — both runs see the full batch.
    gate_kwargs = (args.K, args.d_low, args.t_high)
    cut_mask_full = [gate_fires(g, *gate_kwargs) for g in groups]
    n_cut = sum(cut_mask_full)
    print(f"[trainer] gate (K={args.K}, d<{args.d_low}, term>={args.t_high}) "
          f"would fire on {n_cut}/{len(groups)} groups; "
          f"actually applied: {args.gate}", flush=True)
    cut_mask = cut_mask_full if args.gate == "on" else [False] * len(groups)

    # Pre-compute advantages
    advantages = []
    for g in groups:
        rewards = [t["total_reward"] for t in g["trajectories"]]
        advantages.append(grpo_advantages(rewards))

    # Load model
    print(f"[trainer] loading {args.model} ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    n_gpus = torch.cuda.device_count()
    device_map = "balanced" if n_gpus >= 2 else {"": 0}
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", device_map=device_map,
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
    device = next(model.parameters()).device

    # Indices we may sample from each step
    all_idxs = list(range(len(groups)))

    # Training loop
    for step in range(args.steps):
        t_step_start = time.time()
        # Sample groups
        rng.shuffle(all_idxs)
        batch_ix = all_idxs[: args.groups_per_step]
        # Mark cuts
        is_cut = [cut_mask[i] for i in batch_ix]
        n_groups_cut = sum(is_cut)

        # Build per-(traj, step) training items, EXCLUDING cut groups
        items: List[tuple] = []   # (chat_text, assistant_text, advantage_for_traj)
        for ig, g_ix in enumerate(batch_ix):
            if is_cut[ig]:
                continue  # gate predicted zero-var ⇒ skip
            g = groups[g_ix]
            adv = advantages[g_ix]
            G = len(g["trajectories"])
            for ti in range(G):
                traj = g["trajectories"][ti]
                # Cap how many steps per trajectory we actually train on
                n_steps_avail = len(traj["steps"])
                if n_steps_avail == 0:
                    continue
                # Sample uniformly from available steps to limit token cost
                step_idxs = rng.sample(
                    range(n_steps_avail),
                    k=min(args.max_train_steps_per_traj, n_steps_avail),
                )
                for si in step_idxs:
                    built = build_chat_for_step(g, ti, si)
                    if built is None:
                        continue
                    chat, asst = built
                    chat_text = tok.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True)
                    items.append((chat_text, asst, float(adv[ti])))

        rollout_secs = time.time() - t_step_start  # actually "data prep secs"

        # Forward + backward per item, immediate backward to avoid graph
        # accumulation.
        loss_total = 0.0
        n_items = max(len(items), 1)
        n_zero_var_groups = sum(
            1 for i in batch_ix
            if float(np.var([t["total_reward"] for t in groups[i]["trajectories"]])) == 0.0
        )
        if items:
            optimizer.zero_grad()
            for chat_text, asst_text, adv_val in items:
                full = chat_text + asst_text
                prefix_ids = tok(chat_text, add_special_tokens=False)["input_ids"]
                # Truncate from left to keep recent context
                max_len = 1024
                enc = tok(full, return_tensors="pt", padding=False,
                          truncation=True, max_length=max_len).to(device)
                input_ids = enc["input_ids"]
                if input_ids.size(1) <= len(prefix_ids):
                    continue
                # Compute log p of assistant tokens
                out = model(input_ids=input_ids,
                            attention_mask=enc["attention_mask"])
                logits = out.logits[:, :-1, :]
                labels = input_ids[:, 1:].clone()
                # Build mask: True for assistant-region tokens (positions in
                # the labels view that correspond to the assistant text).
                start = max(0, len(prefix_ids) - 1)
                mask = torch.zeros_like(labels, dtype=torch.bool)
                if start < labels.size(1):
                    mask[:, start:] = (enc["attention_mask"][:, 1:][:, start:] == 1)
                if mask.sum() == 0:
                    continue
                T = logits.size(1)
                chunk = 256
                logp_total = torch.zeros((), device=device, dtype=torch.float32)
                count = 0
                for s in range(0, T, chunk):
                    e = min(s + chunk, T)
                    sub_logits = logits[:, s:e, :]
                    sub_labels = labels[:, s:e]
                    sub_mask = mask[:, s:e]
                    nll = F.cross_entropy(
                        sub_logits.reshape(-1, sub_logits.size(-1)),
                        sub_labels.reshape(-1),
                        reduction="none",
                    ).view_as(sub_labels).float()
                    logp_total = logp_total - (nll * sub_mask).sum()
                    count += int(sub_mask.sum().item())
                    del sub_logits, nll
                if count == 0:
                    continue
                mean_logp = logp_total / count
                # Per-item loss; divide by n_items to keep equivalent of mean.
                loss_b = -(adv_val * mean_logp) / n_items
                loss_b.backward()
                loss_total += float(loss_b.detach()) * n_items
                del out, logits, labels, mask, mean_logp, loss_b, enc, input_ids
                torch.cuda.empty_cache()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0))
            optimizer.step()
        else:
            grad_norm = 0.0
        loss_val = loss_total / n_items if n_items > 0 else 0.0

        # Reward stats for this batch
        rewards_mean = float(np.mean([
            np.mean([t["total_reward"] for t in groups[i]["trajectories"]])
            for i in batch_ix
        ]))

        step_secs = time.time() - t_step_start

        rec = {
            "step": step,
            "gate": args.gate,
            "n_groups": len(batch_ix),
            "n_groups_cut": n_groups_cut if args.gate == "on" else 0,
            "n_zero_variance": n_zero_var_groups,
            "n_train_items": len(items),
            "loss": loss_val,
            "grad_norm": grad_norm,
            "rewards_mean": rewards_mean,
            "rewards_max": float(np.max([
                np.max([t["total_reward"] for t in groups[i]["trajectories"]])
                for i in batch_ix
            ])),
            "rollout_secs": rollout_secs,
            "step_secs": step_secs,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        gate_label = "on" if args.gate == "on" else "off"
        print(f"[step {step:02d}] gate={gate_label}  "
              f"cut={rec['n_groups_cut']}/{len(batch_ix)}  "
              f"zv={n_zero_var_groups}/{len(batch_ix)}  "
              f"items={len(items)}  loss={loss_val:+.4f}  "
              f"gn={grad_norm:.3f}  R̄={rewards_mean:+.3f}  "
              f"step={step_secs:.1f}s", flush=True)

    print(f"[trainer] done. Wrote {log_path}")


if __name__ == "__main__":
    main()
