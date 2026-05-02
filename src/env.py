"""
ALFWorld env wrapper: load a single game, run G parallel instances (same game, different seeds)
so a rollout group shares initial state and diverges only through LLM sampling.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

# Pin the ALFWorld data location BEFORE importing anything from alfworld — its
# `info` module sets os.environ["ALFWORLD_DATA"] at import time from ~/.cache/alfworld
# if the var is unset, which locks us out of the real data dir.
ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA") or "/data1/zhiyuanzhai/selective_rollout/alfworld_data"
os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA

from . import patch_textworld  # noqa: F401,E402  — must import before textworld

import textworld  # noqa: E402
import textworld.gym  # noqa: E402
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos  # noqa: E402

TASK_RE = re.compile(r"your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def list_games(split: str = "valid_seen") -> List[str]:
    pattern = f"{ALFWORLD_DATA}/json_2.1.1/{split}/*/*/game.tw-pddl"
    return sorted(glob.glob(pattern))


@dataclass
class StepResult:
    obs: List[str]          # length G
    reward: List[float]     # length G
    done: List[bool]        # length G
    won: List[bool]         # length G
    admissible: List[List[str]]  # length G


class GroupEnv:
    """G parallel textworld instances of the same game. Once a trajectory is `done`,
    its slot is frozen: `obs` keeps the last observation, `admissible` becomes empty."""

    def __init__(self, game_file: str, group_size: int, max_steps: int = 30,
                 asynchronous: bool = True):
        self.game_file = game_file
        self.G = group_size
        self.max_steps = max_steps
        self._frozen_obs: List[str] = [""] * group_size
        self._frozen_won: List[bool] = [False] * group_size
        self._done_mask: List[bool] = [False] * group_size

        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=["gamefile"]
        )
        # asynchronous=True isolates each slot in its own subprocess (correct
        # but flaky in some environments). Set asynchronous=False as a fallback
        # for setups where multiprocessing fails — the cost is potential state
        # leakage between slots, which is acceptable for the trainer's
        # demonstration runs but not for the 100-task offline analysis.
        env_id = textworld.gym.register_games(
            [game_file] * group_size,
            request_infos,
            batch_size=group_size,
            asynchronous=asynchronous,
            max_episode_steps=max_steps,
            wrappers=wrappers,
        )
        self.env = textworld.gym.make(env_id)

    def reset(self) -> Tuple[str, StepResult]:
        """Return (task_description, initial step result)."""
        obs, infos = self.env.reset()
        obs = list(obs)
        adm = list(infos["admissible_commands"])
        m = TASK_RE.search(obs[0])
        task_desc = m.group(1).strip() if m else "(task not found)"
        self._frozen_obs = list(obs)
        self._frozen_won = [False] * self.G
        self._done_mask = [False] * self.G
        return task_desc, StepResult(
            obs=obs, reward=[0.0] * self.G, done=[False] * self.G,
            won=[False] * self.G, admissible=adm,
        )

    def step(self, commands: List[str]) -> StepResult:
        """Step all G slots. Slots already done receive a harmless no-op ('look')
        and their output is replaced with the frozen last observation."""
        assert len(commands) == self.G
        to_send = [
            "look" if self._done_mask[i] else commands[i] for i in range(self.G)
        ]
        obs, reward, done, infos = self.env.step(to_send)
        obs = list(obs)
        won = list(infos["won"])
        adm = list(infos["admissible_commands"])

        merged_obs, merged_r, merged_done, merged_won, merged_adm = [], [], [], [], []
        for i in range(self.G):
            if self._done_mask[i]:
                merged_obs.append(self._frozen_obs[i])
                merged_r.append(0.0)
                merged_done.append(True)
                merged_won.append(self._frozen_won[i])
                merged_adm.append([])
            else:
                merged_obs.append(obs[i])
                merged_r.append(float(reward[i]))
                merged_done.append(bool(done[i]))
                merged_won.append(bool(won[i]))
                merged_adm.append(adm[i])
                if merged_done[-1]:
                    self._done_mask[i] = True
                    self._frozen_obs[i] = obs[i]
                    self._frozen_won[i] = bool(won[i])
        return StepResult(
            obs=merged_obs, reward=merged_r, done=merged_done,
            won=merged_won, admissible=merged_adm,
        )

    @property
    def all_done(self) -> bool:
        return all(self._done_mask)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
