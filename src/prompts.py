"""
ReAct-style system prompt for ALFWorld (text-only), plus action parsing.

Strategy: one-shot demonstration in the system prompt, then the live task is played
through the chat's user/assistant alternation. This keeps each LLM call cheap and lets
VLLM's prefix cache amortize the system prompt across all G trajectories of a task.
"""
from __future__ import annotations

import re
from typing import List


SYSTEM_PROMPT = """You are a text-based agent solving household tasks.

Rules:
1. Each turn the environment tells you what you see. You reply with exactly ONE action.
2. Format your reply as two lines:
   Thought: <one sentence of reasoning>
   Action: <the action>
3. Pick actions from the admissible list when given; otherwise use short commands like
   "go to <receptacle>", "open <receptacle>", "take <obj> from <recep>",
   "put <obj> in/on <recep>", "use <obj>", "heat <obj> with <recep>",
   "cool <obj> with <recep>", "clean <obj> with <recep>", "examine <obj>", "look", "inventory".
4. Do not invent objects you have not seen. Keep thought short.

Example (different task):

Observation: You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, a drawer 1, a fridge 1, a microwave 1, a sinkbasin 1, a stoveburner 1, a toaster 1.
Your task is to: heat some mug and put it on coffeemachine.
Thought: To heat a mug I first need to find one; mugs are usually on the countertop or in cabinets.
Action: go to countertop 1

Observation: On the countertop 1, you see a mug 1, a pan 1, and a spatula 1.
Thought: The mug is here — take it.
Action: take mug 1 from countertop 1

Observation: You pick up the mug 1 from the countertop 1.
Thought: Now heat it in the microwave.
Action: go to microwave 1

Observation: The microwave 1 is closed.
Thought: Heat the mug using the microwave.
Action: heat mug 1 with microwave 1

Now solve this new task. Respond with exactly one Thought/Action pair per turn."""


ACTION_RE = re.compile(r"^\s*action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
THOUGHT_RE = re.compile(r"^\s*thought\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _fuzzy_pick(cmd: str, admissible: List[str]) -> str | None:
    """Substring / lower-case match; fall back to best Jaccard over tokens."""
    if not admissible:
        return None
    cmd_l = cmd.lower().strip().rstrip(".")
    for a in admissible:
        if a.lower() == cmd_l:
            return a
    for a in admissible:
        if cmd_l in a.lower() or a.lower() in cmd_l:
            return a
    cmd_tokens = set(cmd_l.split())
    if not cmd_tokens:
        return None
    best, best_score = None, 0.0
    for a in admissible:
        at = set(a.lower().split())
        if not at:
            continue
        score = len(cmd_tokens & at) / len(cmd_tokens | at)
        if score > best_score:
            best, best_score = a, score
    return best if best_score >= 0.3 else None


def parse_action(completion: str, admissible: List[str] | None = None) -> tuple[str, str]:
    """Return (action, thought). Action is always one of admissible if that list is provided
    and a reasonable match exists; otherwise falls back to the raw extracted command, or 'look'."""
    thoughts = THOUGHT_RE.findall(completion)
    thought = thoughts[-1].strip() if thoughts else ""
    actions = ACTION_RE.findall(completion)
    raw = actions[-1].strip() if actions else ""
    if not raw:
        # last-ditch: first non-empty line that doesn't start with "thought"
        for line in completion.strip().splitlines():
            s = line.strip()
            if s and not s.lower().startswith("thought"):
                raw = s.lstrip("-*> ").strip()
                break
    raw = raw.rstrip(".").strip()
    if admissible:
        picked = _fuzzy_pick(raw, admissible)
        if picked is not None:
            return picked, thought
    return raw or "look", thought


_OBS_CHAR_LIMIT = 800
_ADMISSIBLE_LIMIT = 25


def _truncate_obs(obs: str) -> str:
    s = obs.strip()
    return s if len(s) <= _OBS_CHAR_LIMIT else s[: _OBS_CHAR_LIMIT - 3] + "..."


def _compact_admissible(admissible: List[str]) -> str:
    # Prioritize non-navigation actions (take/put/open/close/heat/cool/clean/use/examine)
    # over the usually numerous "go to X" / "examine X" options so we keep the
    # most semantically meaningful commands when truncating.
    prio_verbs = ("take ", "put ", "open ", "close ", "heat ",
                  "cool ", "clean ", "use ", "slice ", "move ", "toggle ")
    prio = [a for a in admissible if a.startswith(prio_verbs)]
    rest = [a for a in admissible if a not in prio]
    kept = (prio + rest)[:_ADMISSIBLE_LIMIT]
    return ", ".join(kept) if kept else "(none)"


def initial_user_message(obs: str) -> str:
    # obs already contains "Your task is to: ..."
    return f"Observation: {_truncate_obs(obs)}"


def followup_user_message(obs: str, admissible: List[str]) -> str:
    return f"Observation: {_truncate_obs(obs)}\nAdmissible actions: {_compact_admissible(admissible)}"
