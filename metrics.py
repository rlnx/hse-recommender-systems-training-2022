from typing import Sequence, Set
import pandas as pd

def compute_recsys_metrics(
    predicted: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    user_key = 'user_id',
    item_key = 'item_id'
):
    assert user_key in predicted.columns
    assert item_key in predicted.columns
    assert user_key in test.columns
    assert item_key in test.columns
    assert k > 0

    predicted_grouped = (
        predicted
        .groupby(user_key)
        .agg({item_key: list})
        .rename(columns={item_key: 'predicted'})
    )

    test_grouped = (
        test
        .groupby(user_key)
        .agg({item_key: set})
        .rename(columns={item_key: 'ground_truth'})
    )

    items_to_compare = predicted_grouped.merge(
        test_grouped,
        on=user_key,
        how='left'
    )

    metrics = (
        items_to_compare
        .apply(
            lambda row: _metrics(row['predicted'], row['ground_truth'], k),
            axis=1,
            result_type='expand'
        )
        .rename(columns={ 0: 'recall', 1: 'map' })
    )

    return metrics.mean().to_dict()


def _metrics(predicted: Sequence, ground_truth: Set, k: int):
    if not ground_truth:
        return 0.0, 0.0

    predicted_k = predicted[:k]

    # Recall@k
    intersection = ground_truth.intersection(predicted_k)
    recall = len(intersection) / min(len(ground_truth), len(predicted_k))

    # Average Precision (AP@k)
    num_hits = 0.0
    ap_sum = 0.0
    for i, pr in enumerate(predicted_k):
        if pr in ground_truth and pr not in predicted[:i]:
            num_hits += 1
            ap_sum += num_hits / (i + 1.0)
    ap_score = ap_sum / min(len(ground_truth), len(predicted_k))

    return recall, ap_score
