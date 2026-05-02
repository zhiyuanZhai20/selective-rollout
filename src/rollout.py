"""
Per-task group rollout: G parallel trajectories on the same ALFWorld game.

All G slots are stepped in lockstep: at every turn we build G chat histories, batch
them through VLLM in one call, parse G actions, and step the GroupEnv. We record
(observation, thought, action, reward, done) per step per trajectory.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any

from .env import GroupEnv
from .llm import ChatLLM
from .prompts import (
    SYSTEM_PROMPT,
    initial_user_message,
    followup_user_message,
    parse_action,
)


@dataclass
class StepRecord:
    obs: str
    thought: str
    action: str
    raw_completion: str
    reward: float
    done: bool
    won: bool


@dataclass
class Trajectory:
    steps: List[StepRecord] = field(default_factory=list)
    total_reward: float = 0.0
    won: bool = False
    terminated_at: int = -1  # step idx at which done became True; -1 means not terminated


@dataclass
class GroupRollout:
    task_id: str
    task_desc: str
    game_file: str
    group_size: int
    max_steps: int
    trajectories: List[Trajectory]
    wall_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_desc": self.task_desc,
            "game_file": self.game_file,
            "group_size": self.group_size,
            "max_steps": self.max_steps,
            "wall_time_sec": self.wall_time_sec,
            "trajectories": [
                {
                    "total_reward": t.total_reward,
                    "won": t.won,
                    "terminated_at": t.terminated_at,
                    "steps": [asdict(s) for s in t.steps],
                }
                for t in self.trajectories
            ],
        }


def rollout_group(
    llm: ChatLLM,
    game_file: str,
    task_id: str,
    group_size: int = 8,
    max_steps: int = 30,
    temperature: float = 0.7,
    max_new_tokens: int = 96,
    gate: Dict[str, float] | None = None,
) -> GroupRollout:
    """Run G trajectories of the same game to completion (or max_steps).

    If `gate` is supplied (e.g. {"K": 10, "d_low": 0.12, "t_high": 0.90}), the
    rollout evaluates the OR-rule selective-rollout gate at step `K`: when the
    rule fires, all alive trajectories are terminated immediately and the
    group's wall-clock advantage is realised.
    """
    from .divergence import divergence_at_K  # local import to avoid cycles
    t0 = time.time()
    env = GroupEnv(game_file, group_size=group_size, max_steps=max_steps)
    task_desc, sr = env.reset()
    trajectories = [Trajectory() for _ in range(group_size)]

    # Per-slot chat history. All start with the same system + initial user msg
    # (shared prefix → free from VLLM's prefix cache).
    chats: List[List[Dict[str, str]]] = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": followup_user_message(sr.obs[i], sr.admissible[i])},
        ]
        for i in range(group_size)
    ]

    # Latest observation shown to the model for each slot (the user message content).
    # Used to attribute "obs at step t" correctly in StepRecord.
    last_obs_per_slot = list(sr.obs)

    for step_idx in range(max_steps):
        if env.all_done:
            break

        # Only run LLM for slots that are still alive; skip dead slots.
        alive = [i for i in range(group_size) if trajectories[i].terminated_at == -1]
        if not alive:
            break

        alive_chats = [chats[i] for i in alive]
        completions_alive = llm.generate(
            alive_chats,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        completions: List[str] = [""] * group_size
        for k, i in enumerate(alive):
            completions[i] = completions_alive[k]

        # Parse actions per slot
        actions: List[str] = []
        thoughts: List[str] = []
        for i in range(group_size):
            if trajectories[i].terminated_at != -1:
                actions.append("look")  # dummy, env will ignore
                thoughts.append("")
            else:
                a, th = parse_action(completions[i], admissible=sr.admissible[i])
                actions.append(a)
                thoughts.append(th)

        # Step env
        sr_next = env.step(actions)

        # Record step + extend chat history
        for i in range(group_size):
            if trajectories[i].terminated_at != -1:
                continue
            trajectories[i].steps.append(
                StepRecord(
                    obs=last_obs_per_slot[i],
                    thought=thoughts[i],
                    action=actions[i],
                    raw_completion=completions[i],
                    reward=sr_next.reward[i],
                    done=sr_next.done[i],
                    won=sr_next.won[i],
                )
            )
            trajectories[i].total_reward += sr_next.reward[i]
            # assistant turn (what we sent back)
            chats[i].append(
                {"role": "assistant", "content": f"Thought: {thoughts[i]}\nAction: {actions[i]}"}
            )
            # next user turn (new observation)
            if sr_next.done[i]:
                trajectories[i].terminated_at = step_idx
                trajectories[i].won = sr_next.won[i]
            else:
                chats[i].append(
                    {"role": "user",
                     "content": followup_user_message(sr_next.obs[i], sr_next.admissible[i])}
                )
                last_obs_per_slot[i] = sr_next.obs[i]

        sr = sr_next  # for admissible list on next iter

        # Selective-rollout gate: at step K, evaluate the OR rule and stop the
        # whole group if it fires. We mark every still-alive trajectory as
        # terminated_at=step_idx so downstream analysis can tell this group was
        # cut by the gate (vs naturally finished).
        if gate is not None and step_idx + 1 == int(gate["K"]):
            actions_per_traj = [
                [s.action for s in trajectories[i].steps]
                for i in range(group_size)
            ]
            obs_per_traj = [
                [s.obs for s in trajectories[i].steps]
                for i in range(group_size)
            ]
            metrics = divergence_at_K(actions_per_traj, obs_per_traj,
                                      K=int(gate["K"]))
            d_K = metrics["prefix_edit_distance_mean"]
            t_K = metrics["termination_fraction"]
            if (d_K < gate["d_low"]) or (t_K >= gate["t_high"]):
                for i in range(group_size):
                    if trajectories[i].terminated_at == -1:
                        trajectories[i].terminated_at = step_idx
                # Break out of the for-loop ⇒ rollout ends here.
                break

    env.close()
    return GroupRollout(
        task_id=task_id,
        task_desc=task_desc,
        game_file=game_file,
        group_size=group_size,
        max_steps=max_steps,
        trajectories=trajectories,
        wall_time_sec=time.time() - t0,
    )


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
