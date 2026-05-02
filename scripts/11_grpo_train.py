"""
Phase C.1/C.2 — minimal GRPO trainer for ALFWorld with optional selective-rollout gate.

Design choices for fitting on a single 48GB GPU:
  * Qwen2.5-7B-Instruct + LoRA (r=16, q_proj/v_proj only)
  * HF transformers.generate() for rollout; no VLLM (avoids LoRA hot-swap pain)
  * No KV cache reuse across env steps (HF handles each turn fresh)
  * Group size G=4 (smaller than offline G=8) to keep step compute bounded
  * Token-level GRPO loss with whitened group-relative advantages
  * SINGLE optimizer step per training step

Usage:
    python scripts/11_grpo_train.py --gate off --steps 20 --out runs/baseline
    python scripts/11_grpo_train.py --gate on  --steps 20 --out runs/gated
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import GroupEnv, list_games  # noqa: E402
from src.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    followup_user_message,
    parse_action,
)
from src.divergence import divergence_at_K  # noqa: E402


@dataclass
class TrainStep:
    """Per-(prompt, traj, env-step) record needed by the GRPO loss."""
    group_id: int
    traj_id: int
    step_idx: int
    chat_text_before_assistant: str  # full chat text up through the user msg before assistant turn
    assistant_text: str              # what the model generated, will be tokenised + reused
    advantage: float                 # group-relative advantage of this trajectory
    # Filled in during the loss computation:
    n_tokens_assistant: int = 0


# ---------- HF batched generate ----------
class HFGenerator:
    """Wraps a peft-wrapped causal LM with a batched generate() that mirrors
    src.llm.ChatLLM.generate's interface but runs through HF transformers."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(
        self,
        chats: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 96,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Batched generate. Returns one completion per chat, truncated at the
        first occurrence of any stop string (best-effort post-decoding)."""
        prompts = [
            self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            for chat in chats
        ]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=4096).to(self.model.device)
        gen = self.model.generate(
            **enc,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Decode only the new tokens (after the prompt for each row).
        completions = []
        for i in range(gen.size(0)):
            in_len = enc["input_ids"][i].size(0)
            new_ids = gen[i, in_len:]
            # Strip pad tokens after EOS.
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            if stop:
                for s in stop:
                    j = text.find(s)
                    if j != -1:
                        text = text[:j]
                        break
            completions.append(text)
        return completions


# ---------- group rollout (training-time, with optional mid-rollout gate) ----------
@dataclass
class TrainGroupRollout:
    task_id: str
    group_size: int
    max_steps: int
    actions_per_traj: List[List[str]]
    rewards_per_traj: List[float]   # terminal reward per trajectory
    cut_at_step: int                # -1 = not cut; otherwise the step at which we cut
    train_steps: List[TrainStep]    # only kept-trajectory, kept-step records
    wall_time_sec: float


def rollout_group_for_training(
    gen: HFGenerator,
    game_file: str,
    task_id: str,
    group_size: int = 4,
    max_steps: int = 30,
    temperature: float = 0.7,
    max_new_tokens: int = 96,
    gate: Optional[Dict[str, float]] = None,  # {"K": 10, "d_low": 0.12, "t_high": 0.90}
) -> TrainGroupRollout:
    t0 = time.time()
    env = GroupEnv(game_file, group_size=group_size, max_steps=max_steps,
                    asynchronous=False)
    task_desc, sr = env.reset()

    chats: List[List[Dict[str, str]]] = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": followup_user_message(sr.obs[i], sr.admissible[i])},
        ]
        for i in range(group_size)
    ]
    actions_per_traj: List[List[str]] = [[] for _ in range(group_size)]
    rewards: List[float] = [0.0] * group_size
    terminated_at: List[int] = [-1] * group_size
    won: List[bool] = [False] * group_size
    train_steps: List[TrainStep] = []
    cut_at_step = -1

    for step_idx in range(max_steps):
        if env.all_done:
            break
        alive = [i for i in range(group_size) if terminated_at[i] == -1]
        if not alive:
            break

        alive_chats = [chats[i] for i in alive]
        completions_alive = gen.generate(
            alive_chats,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=["\nObservation:", "\nObservation ", "<|im_end|>"],
        )

        completions: List[str] = [""] * group_size
        for k, i in enumerate(alive):
            completions[i] = completions_alive[k]

        # Capture (chat-before-assistant, assistant-text) before mutating chats.
        prompt_texts: List[str] = [
            gen.tokenizer.apply_chat_template(chats[i], tokenize=False,
                                              add_generation_prompt=True)
            for i in range(group_size)
        ]
        actions: List[str] = []
        thoughts: List[str] = []
        for i in range(group_size):
            if terminated_at[i] != -1:
                actions.append("look"); thoughts.append("")
                continue
            a, th = parse_action(completions[i], admissible=sr.admissible[i])
            actions.append(a); thoughts.append(th)
            actions_per_traj[i].append(a)

        sr_next = env.step(actions)

        # Per-(traj, step) bookkeeping
        for i in range(group_size):
            if terminated_at[i] != -1:
                continue
            assistant_text = f"Thought: {thoughts[i]}\nAction: {actions[i]}"
            train_steps.append(TrainStep(
                group_id=-1, traj_id=i, step_idx=step_idx,
                chat_text_before_assistant=prompt_texts[i],
                assistant_text=assistant_text,
                advantage=0.0,  # filled later
            ))
            rewards[i] += sr_next.reward[i]
            chats[i].append({"role": "assistant", "content": assistant_text})
            if sr_next.done[i]:
                terminated_at[i] = step_idx
                won[i] = sr_next.won[i]
            else:
                chats[i].append({
                    "role": "user",
                    "content": followup_user_message(sr_next.obs[i],
                                                      sr_next.admissible[i]),
                })
        sr = sr_next

        # Mid-rollout gate evaluation
        if gate is not None and step_idx + 1 == int(gate["K"]):
            obs_per_traj = [["" for _ in actions_per_traj[i]] for i in range(group_size)]
            metrics = divergence_at_K(actions_per_traj, obs_per_traj, K=int(gate["K"]))
            d_K = metrics["prefix_edit_distance_mean"]
            t_K = metrics["termination_fraction"]
            cut = (d_K < gate["d_low"]) or (t_K >= gate["t_high"])
            if cut:
                cut_at_step = step_idx + 1
                env.close()
                # Drop all train_steps after K (none of them happened, but for
                # the "post-cut" semantics we *do* keep the first K steps and
                # use them in loss only if the group's terminal reward variance
                # turns out non-zero. For consistency we predict zero-variance
                # ⇒ we skip this whole group from the loss.)
                return TrainGroupRollout(
                    task_id=task_id,
                    group_size=group_size,
                    max_steps=max_steps,
                    actions_per_traj=actions_per_traj,
                    rewards_per_traj=rewards,
                    cut_at_step=cut_at_step,
                    train_steps=[],  # empty: gate predicts zero-variance, skip group
                    wall_time_sec=time.time() - t0,
                )

    env.close()
    return TrainGroupRollout(
        task_id=task_id,
        group_size=group_size,
        max_steps=max_steps,
        actions_per_traj=actions_per_traj,
        rewards_per_traj=rewards,
        cut_at_step=cut_at_step,
        train_steps=train_steps,
        wall_time_sec=time.time() - t0,
    )


# ---------- GRPO loss ----------
def compute_grpo_loss_and_backward(
    model,
    tokenizer,
    train_steps: List[TrainStep],
    advantages: torch.Tensor,  # one scalar per train_step (already tied)
    micro_batch: int = 1,
) -> float:
    """Token-level policy-gradient loss; immediate backward per micro-batch
    to avoid accumulating live autograd graphs across the loop.

    Returns the (detached) total loss as a float for logging.
    """
    device = next(model.parameters()).device
    total_loss_val = 0.0
    n_steps_total = max(len(train_steps), 1)
    for batch_start in range(0, len(train_steps), micro_batch):
        batch = train_steps[batch_start: batch_start + micro_batch]
        adv_batch = advantages[batch_start: batch_start + micro_batch]

        # Build full strings = prefix + assistant
        full_strs, prefix_lens = [], []
        for ts in batch:
            full = ts.chat_text_before_assistant + ts.assistant_text
            full_strs.append(full)
            prefix_ids = tokenizer(ts.chat_text_before_assistant,
                                    add_special_tokens=False)["input_ids"]
            prefix_lens.append(len(prefix_ids))
        enc = tokenizer(full_strs, return_tensors="pt", padding=True,
                        truncation=True, max_length=4096).to(device)

        # Truncate aggressively from the LEFT to keep recent context only.
        max_len = 1024
        if enc["input_ids"].size(1) > max_len:
            enc["input_ids"] = enc["input_ids"][:, -max_len:]
            enc["attention_mask"] = enc["attention_mask"][:, -max_len:]
            # adjust prefix_lens to be relative to the truncated start
            new_prefix_lens = []
            cut = enc["input_ids"].size(1)  # length after truncation
            full_len_before = max_len + (max_len - cut)  # not used; placeholder
            for plen in prefix_lens:
                # If prefix was 800 tokens but we truncated start, the new prefix
                # length might be 0; treat as 0 (whole window is assistant region).
                new_prefix_lens.append(max(0, plen - (max_len - cut)))
            prefix_lens = new_prefix_lens

        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"])
        logits = out.logits[:, :-1, :]  # predict next token
        labels = enc["input_ids"][:, 1:].clone()
        attn = enc["attention_mask"][:, 1:]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, plen in enumerate(prefix_lens):
            start = max(plen - 1, 0)
            mask[i, start:] = (attn[i, start:] == 1)

        # Memory-efficient log-prob computation: chunk along the sequence axis
        # so we never materialise the full (B, T, V) log_softmax tensor.
        T = logits.size(1)
        chunk = 256
        per_row_logp = torch.zeros(logits.size(0), device=logits.device,
                                    dtype=torch.float32)
        per_row_count = torch.zeros_like(per_row_logp)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            sub_logits = logits[:, s:e, :]   # keep bf16 to save memory
            sub_labels = labels[:, s:e]
            sub_mask = mask[:, s:e]
            nll = F.cross_entropy(
                sub_logits.reshape(-1, sub_logits.size(-1)),
                sub_labels.reshape(-1),
                reduction="none",
            ).view_as(sub_labels).float()
            per_row_logp = per_row_logp - (nll * sub_mask).sum(dim=1)
            per_row_count = per_row_count + sub_mask.sum(dim=1).float()
            del sub_logits, nll
        per_row = per_row_logp / per_row_count.clamp(min=1)
        # PG loss: maximise A·logp ⇒ minimise -A·logp
        # Divide by total micro-batches so the equivalent of summing then
        # dividing is preserved across the loop.
        loss_b = -(adv_batch * per_row).sum() / n_steps_total
        loss_b.backward()
        total_loss_val += float(loss_b.detach()) * n_steps_total
        # Free the autograd graph + intermediates immediately.
        del out, logits, per_row_logp, per_row_count, per_row, loss_b
        del enc, labels, mask, attn
        torch.cuda.empty_cache()
    return total_loss_val / n_steps_total


# ---------- main training loop ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gate", choices=["on", "off"], default="off")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--d_low", type=float, default=0.12)
    ap.add_argument("--t_high", type=float, default=0.90)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-rollout-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="runs/grpo")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.jsonl"

    # Load model + LoRA (BF16 base + bf16 LoRA, multi-GPU via device_map="auto")
    print(f"[trainer] loading {args.model} (bf16, multi-GPU) ...", flush=True)
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        # Force balanced split across visible GPUs to avoid OOM on GPU 0.
        max_memory = {i: "20GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "16GiB"
        device_map = "balanced"
    else:
        max_memory = None
        device_map = {"": 0}
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
        device_map=device_map,
        max_memory=max_memory,
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
    model.config.use_cache = False
    gc.collect(); torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    gen = HFGenerator(model, tok)

    # Game pool
    games = list_games(args.split)
    rng = random.Random(args.seed)
    rng.shuffle(games)
    print(f"[trainer] using {len(games)} games from split={args.split}")

    gate_dict = ({"K": args.K, "d_low": args.d_low, "t_high": args.t_high}
                 if args.gate == "on" else None)

    # Training loop
    step_log = []
    for step in range(args.steps):
        t_step_start = time.time()
        # Sample N prompts
        prompts = [games[(step * args.prompts_per_step + j) % len(games)]
                   for j in range(args.prompts_per_step)]
        # Rollout each
        rgs: List[TrainGroupRollout] = []
        rollout_t0 = time.time()
        for gi, gf in enumerate(prompts):
            tid = "/".join(gf.split("/")[-3:-1])
            rg = rollout_group_for_training(
                gen, gf, tid,
                group_size=args.group_size,
                max_steps=args.max_rollout_steps,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                gate=gate_dict,
            )
            for ts in rg.train_steps:
                ts.group_id = gi
            rgs.append(rg)
        rollout_secs = time.time() - rollout_t0

        # Aggregate stats
        all_train_steps: List[TrainStep] = []
        all_advantages: List[float] = []
        n_groups_total = len(rgs)
        n_groups_cut = sum(1 for rg in rgs if rg.cut_at_step != -1)
        n_zero_var = sum(1 for rg in rgs if np.var(rg.rewards_per_traj) == 0)
        rewards_mean = float(np.mean([np.mean(rg.rewards_per_traj) for rg in rgs]))
        rewards_max = float(np.max([np.max(rg.rewards_per_traj) for rg in rgs]))

        for rg in rgs:
            if rg.cut_at_step != -1:
                continue  # gate predicted zero-variance: contribute no loss
            r = np.array(rg.rewards_per_traj)
            adv = (r - r.mean()) / (r.std() + 1e-8)
            for ts in rg.train_steps:
                ts.advantage = float(adv[ts.traj_id])
            all_train_steps.extend(rg.train_steps)
            all_advantages.extend([ts.advantage for ts in rg.train_steps])

        if not all_train_steps:
            print(f"[step {step:02d}] no train steps (all groups cut or zero-var) — skipping update")
            grad_norm = 0.0
            loss_val = 0.0
        else:
            import gc as _gc
            _gc.collect(); torch.cuda.empty_cache()
            adv_t = torch.tensor(all_advantages, dtype=torch.float32,
                                  device=model.device)
            optimizer.zero_grad()
            loss_val = compute_grpo_loss_and_backward(
                model, tok, all_train_steps, adv_t, micro_batch=1)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0))
            optimizer.step()
            _gc.collect(); torch.cuda.empty_cache()

        step_secs = time.time() - t_step_start
        rec = {
            "step": step,
            "gate": args.gate,
            "n_groups": n_groups_total,
            "n_groups_cut": n_groups_cut,
            "n_zero_variance": n_zero_var,
            "n_train_steps": len(all_train_steps),
            "loss": loss_val,
            "grad_norm": grad_norm,
            "rewards_mean": rewards_mean,
            "rewards_max": rewards_max,
            "rollout_secs": rollout_secs,
            "step_secs": step_secs,
        }
        step_log.append(rec)
        print(f"[step {step:02d}] gate={args.gate}  cut={n_groups_cut}/{n_groups_total}  "
              f"zv={n_zero_var}/{n_groups_total}  ts={len(all_train_steps)}  "
              f"loss={loss_val:+.4f}  gn={grad_norm:.3f}  "
              f"R̄={rewards_mean:+.3f}  rollout={rollout_secs:.1f}s  step={step_secs:.1f}s",
              flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[trainer] done. Wrote {log_path}")


if __name__ == "__main__":
    main()
