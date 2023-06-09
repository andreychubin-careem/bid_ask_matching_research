import pandas as pd
import numpy as np


def get_cost_matrix(path: str = '_data/sample_snapshot_2.parquet', dropout: float = 0.0) -> np.ndarray:
    data = pd.read_parquet(path)
    dist_matrix = pd.pivot_table(data, values='distance', index='userid', columns='driver_id').values

    if dropout > 0.0:
        dist_matrix = drop_captains(dist_matrix, dropout)

    return dist_matrix


def get_waiting(matrix: np.ndarray) -> (float, float):
    waiting_customers = len(np.where(matrix.sum(axis=1) == 0.0)[0])
    waiting_captains = len(np.where(matrix.sum(axis=0) == 0.0)[0])
    return waiting_customers, waiting_captains


def get_distance_stats_to_customer(matrix: np.ndarray, assignments: np.ndarray) -> float:
    sub = matrix * assignments
    sub[np.isnan(sub)] = 0.0
    sub[np.isinf(sub)] = 0.0
    result = np.true_divide(sub.sum(axis=1), (sub != 0).sum(axis=1))
    result = result[~np.isnan(result)]
    return result.mean()


def drop_captains(matrix: np.ndarray, dropout: float) -> np.ndarray:
    if dropout > 0.0:
        columns = np.arange(0, matrix.shape[1])
        chosen_columns = np.random.choice(columns, size=int((1.0 - dropout) * len(columns)), replace=False)
        chosen_columns.sort()
        return matrix[:, chosen_columns]
    else:
        return matrix


def get_choice_stats(matrix: np.ndarray) -> dict:
    stat = matrix.sum(axis=1)
    quantiles = np.quantile(stat, [0.25, 0.5, 0.75, 0.99])
    result = dict()
    result['mean'] = stat.mean()
    result['q_25'] = quantiles[0]
    result['median'] = quantiles[1]
    result['q_75'] = quantiles[2]
    result['q_99'] = quantiles[3]
    return result
