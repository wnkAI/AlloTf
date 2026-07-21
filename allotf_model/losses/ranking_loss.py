"""Within-TF pairwise ranking: functional sensor > binder-only > dead mutant.

Absolute fold-change is not comparable across datasets/assays, so the supervision that transfers is
the ORDER within one TF. For every pair in the same TF whose tiers differ, the higher tier must score
higher by a margin.
"""
import torch


def pairwise_ranking(scores, tf_ids, tiers, margin=0.1):
    """scores: (B,) S_final per sample. tf_ids: list[str]. tiers: (B,) int (higher = more functional).
    -> scalar hinge loss over same-TF, different-tier pairs."""
    n = len(scores)
    loss = scores.new_zeros(())
    count = 0
    for a in range(n):
        for b in range(n):
            if a == b or tf_ids[a] != tf_ids[b]:
                continue
            if tiers[a] > tiers[b]:
                loss = loss + torch.relu(margin - (scores[a] - scores[b]))
                count += 1
    return loss / count if count else loss
