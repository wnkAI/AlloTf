"""Within-TF pairwise ranking: functional sensor > binder-only > dead mutant.

Absolute fold-change is not comparable across datasets/assays, so the supervision that transfers is
the ORDER within one TF. For every pair in the same TF whose tiers differ, the higher tier must score
higher by a margin. Samples with no TF id or no tier are EXCLUDED (not merged into a phantom group).
"""
import torch


def pairwise_ranking(scores, tf_ids, tiers, margin=0.1):
    """scores: (B,). tf_ids: list (None = exclude). tiers: list of int (None = exclude).
    -> scalar hinge loss over same-TF, different-tier pairs."""
    idx = [k for k in range(len(scores))
           if tf_ids[k] is not None and tiers[k] is not None]
    loss = scores.new_zeros(())
    count = 0
    for a in idx:
        for b in idx:
            if a == b or tf_ids[a] != tf_ids[b] or tiers[a] <= tiers[b]:
                continue
            loss = loss + torch.relu(margin - (scores[a] - scores[b]))
            count += 1
    return loss / count if count else loss
