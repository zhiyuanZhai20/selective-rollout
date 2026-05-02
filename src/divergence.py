"""
Divergence metrics over a group of G trajectories, evaluated at step K.

A trajectory is a list of action strings (observations are also available).
For trajectories that terminated before step K we pad with a special END token
so that n-gram/edit-distance metrics still make sense; but we also expose
`termination_fraction` which says how much of the group has already finished.

Input convention for this module:
- `actions_per_traj`: List[List[str]]  (length G, each inner list is that traj's
  full action sequence up to termination, in order)
- `obs_per_traj`: List[List[str]]      (same shape, observations the agent saw
  at each step — obs[t] is the observation the agent acted on to produce
  actions[t])
- `K`: int prefix length to consider
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence

END = "<END>"


def _prefix(seq: Sequence[str], K: int) -> List[str]:
    """Action prefix of length K, padded with END if the traj ended earlier."""
    p = list(seq[:K])
    while len(p) < K:
        p.append(END)
    return p


def _normalized_edit_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Levenshtein on sequence-of-tokens, normalized to [0, 1] by max length."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[la][lb] / max(la, lb)


def _bigrams(seq: Sequence[str]) -> set:
    return {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)} if len(seq) >= 2 else set()


def _jaccard_dist(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def divergence_at_K(
    actions_per_traj: List[List[str]],
    obs_per_traj: List[List[str]],
    K: int,
) -> Dict[str, float]:
    """Compute all divergence metrics for this group at step K.

    Returns a dict of metric_name -> scalar in [0, 1] (most of them).
    """
    G = len(actions_per_traj)
    assert G == len(obs_per_traj)
    assert G >= 2, "need at least 2 trajectories to measure divergence"

    # Padded prefixes
    prefixes = [_prefix(a, K) for a in actions_per_traj]

    # 1. unique_action_ratio: unique elements at position K-1 (the "current" action)
    actions_at_K = [p[K - 1] for p in prefixes]
    unique_action_ratio = len(set(actions_at_K)) / G

    # 2. unique_prefix_ratio: unique full prefix sequences
    unique_prefix_ratio = len({tuple(p) for p in prefixes}) / G

    # 3. action_bigram_jaccard_mean: mean pairwise Jaccard distance on bigrams
    bigrams = [_bigrams(p) for p in prefixes]
    if G >= 2:
        pairs = list(combinations(range(G), 2))
        bigram_jacc = sum(_jaccard_dist(bigrams[i], bigrams[j]) for i, j in pairs) / len(pairs)
    else:
        bigram_jacc = 0.0

    # 4. prefix_edit_distance_mean: mean pairwise normalized Levenshtein
    edit_dist = sum(
        _normalized_edit_distance(prefixes[i], prefixes[j])
        for i, j in pairs
    ) / len(pairs)

    # 5. obs_unique_ratio at step K-1 (the obs the agent sees when choosing action K)
    obs_at_K = []
    for o in obs_per_traj:
        if K - 1 < len(o):
            obs_at_K.append(o[K - 1])
        else:
            obs_at_K.append(END)
    obs_unique_ratio = len(set(obs_at_K)) / G

    # 6. termination_fraction: how many trajs have already ended by step K
    term_count = sum(1 for a in actions_per_traj if len(a) <= K - 1)
    termination_fraction = term_count / G

    # 7. action_entropy: Shannon entropy of action distribution at position K-1 (normalized)
    import math
    from collections import Counter

    cnt = Counter(actions_at_K)
    total = sum(cnt.values())
    probs = [c / total for c in cnt.values()]
    H = -sum(p * math.log2(p) for p in probs) if probs else 0.0
    H_norm = H / math.log2(G) if G > 1 else 0.0  # max entropy = log2(G)
    action_entropy = H_norm

    return {
        "unique_action_ratio": unique_action_ratio,
        "unique_prefix_ratio": unique_prefix_ratio,
        "action_bigram_jaccard_mean": bigram_jacc,
        "prefix_edit_distance_mean": edit_dist,
        "obs_unique_ratio": obs_unique_ratio,
        "termination_fraction": termination_fraction,
        "action_entropy": action_entropy,
    }


def group_reward_variance(rewards: Sequence[float]) -> float:
    """Final reward variance across a group (population, not sample)."""
    n = len(rewards)
    if n == 0:
        return 0.0
    mu = sum(rewards) / n
    return sum((r - mu) ** 2 for r in rewards) / n


def group_reward_stats(rewards: Sequence[float]) -> Dict[str, float]:
    n = len(rewards)
    mu = sum(rewards) / n if n else 0.0
    var = group_reward_variance(rewards)
    return {
        "reward_mean": mu,
        "reward_variance": var,
        "is_zero_variance": float(var == 0.0),
    }
