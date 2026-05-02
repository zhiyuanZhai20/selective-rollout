# Gate experiment — result

- Tasks rolled out: **22**
- Mean success rate: **0.34**
- Group composition: all-fail **6** / mixed **14** / all-succeed **2** → zero-variance fraction **8/22 (36%)**

## Verdict: **PASS**  best |ρ|=0.458  best AUROC=0.839

Pass criterion: max(|ρ|, 2·(AUROC − 0.5)) ≥ 0.4 on any (metric, K) pair.
AUROC ≥ 0.7 ≈ strong threshold signal; AUROC ≈ 0.5 → no signal.

## Top (metric, K) pairs by |Spearman ρ|

| K | metric | n | ρ | p | AUROC |
| --- | --- | --- | --- | --- | --- |
| 10 | action_bigram_jaccard_mean | 22 | 0.458 | 0.032 | 0.839 |
| 10 | unique_prefix_ratio | 22 | 0.438 | 0.041 | 0.835 |
| 20 | action_bigram_jaccard_mean | 22 | 0.436 | 0.043 | 0.768 |
| 15 | action_bigram_jaccard_mean | 22 | 0.399 | 0.066 | 0.795 |
| 5 | action_bigram_jaccard_mean | 22 | 0.364 | 0.096 | 0.781 |

## Top (metric, K) pairs by AUROC (non-zero-var classification)

| K | metric | n | AUROC | ρ |
| --- | --- | --- | --- | --- |
| 10 | action_bigram_jaccard_mean | 22 | 0.839 | 0.458 |
| 10 | unique_prefix_ratio | 22 | 0.835 | 0.438 |
| 15 | prefix_edit_distance_mean | 22 | 0.817 | 0.326 |
| 10 | obs_unique_ratio | 22 | 0.795 | 0.288 |
| 10 | action_entropy | 22 | 0.795 | 0.243 |

## Strongest signal per task type (best metric × K per type)

| task_type | n | K | metric | ρ | AUROC |
| --- | --- | --- | --- | --- | --- |
| pick_heat_then_place_in_recep | 5 | 10 | action_entropy | 0.803 | 1.000 |
| pick_two_obj_and_place | 5 | 20 | termination_fraction | 1.000 | 1.000 |