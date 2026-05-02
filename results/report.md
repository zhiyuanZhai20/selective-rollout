# Gate experiment — result

- Tasks rolled out: **100**
- Mean success rate: **0.44**
- Group composition: all-fail **25** / mixed **61** / all-succeed **14** → zero-variance fraction **39/100 (39%)**

## Verdict: **PASS**  best |ρ|=0.419  best AUROC=0.773

Pass criterion: max(|ρ|, 2·(AUROC − 0.5)) ≥ 0.4 on any (metric, K) pair.
AUROC ≥ 0.7 ≈ strong threshold signal; AUROC ≈ 0.5 → no signal.

## Top (metric, K) pairs by |Spearman ρ|

| K | metric | n | ρ | p | AUROC |
| --- | --- | --- | --- | --- | --- |
| 15 | prefix_edit_distance_mean | 100 | 0.419 | 1.4e-05 | 0.773 |
| 20 | prefix_edit_distance_mean | 100 | 0.418 | 1.5e-05 | 0.753 |
| 10 | action_bigram_jaccard_mean | 100 | 0.407 | 2.7e-05 | 0.770 |
| 15 | action_bigram_jaccard_mean | 100 | 0.406 | 2.8e-05 | 0.752 |
| 10 | unique_prefix_ratio | 100 | 0.398 | 4e-05 | 0.751 |

## Top (metric, K) pairs by AUROC (non-zero-var classification)

| K | metric | n | AUROC | ρ |
| --- | --- | --- | --- | --- |
| 15 | prefix_edit_distance_mean | 100 | 0.773 | 0.419 |
| 10 | action_bigram_jaccard_mean | 100 | 0.770 | 0.407 |
| 10 | prefix_edit_distance_mean | 100 | 0.757 | 0.374 |
| 20 | prefix_edit_distance_mean | 100 | 0.753 | 0.418 |
| 15 | action_bigram_jaccard_mean | 100 | 0.752 | 0.406 |

## Strongest signal per task type (best metric × K per type)

| task_type | n | K | metric | ρ | AUROC |
| --- | --- | --- | --- | --- | --- |
| look_at_obj_in_light | 8 | 15 | unique_prefix_ratio | 0.810 | 1.000 |
| pick_and_place_simple | 24 | 15 | unique_action_ratio | 0.688 | 0.857 |
| pick_clean_then_place_in_recep | 18 | 15 | termination_fraction | 0.457 | 0.708 |
| pick_cool_then_place_in_recep | 19 | 20 | termination_fraction | 0.730 | 0.821 |
| pick_heat_then_place_in_recep | 11 | 10 | action_entropy | 0.546 | 1.000 |
| pick_two_obj_and_place | 20 | 20 | termination_fraction | 0.788 | 0.857 |