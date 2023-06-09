import numpy as np
from typing import Callable

from .solver.acceptance.captains import greedy_acceptance
from .solver.acceptance.customers import greedy_handshake
from .utils.functions import get_waiting, get_distance_stats_to_customer, drop_captains, get_choice_stats


def wrapper(
        snapshot: str,
        matrix: np.ndarray,
        m: int,
        capacity: int,
        dr: float,
        dropout: float,
        n_possible: int,
        matching_fn: Callable
) -> dict:
    """
    A wrapper function for parallel execution
    :param dropout:
    :param snapshot: path to file
    :param matrix:
    :param m:
    :param capacity:
    :param dr:
    :param n_possible:
    :param matching_fn:
    :return:
    """
    np.random.seed(123)
    drivers_dropout = 0.00
    customers_dropout = 0.00

    if dropout > 0.0:
        matrix = drop_captains(matrix, dropout)

    assignments = matching_fn(matrix, m=m, capacity=capacity, dr=dr)
    idle_c, idle_d = get_waiting(assignments)
    m_dist = get_distance_stats_to_customer(matrix, assignments)
    acceptance = greedy_acceptance(matrix, assignments, n_possible=n_possible, dropout=drivers_dropout)
    idle_c_a, _ = get_waiting(acceptance)
    # a_stats = get_choice_stats(acceptance)
    handshakes = greedy_handshake(matrix, acceptance, dropout=customers_dropout)
    idle_c_h, _ = get_waiting(handshakes)

    return {
        'datetime': snapshot.replace('snapshot_', '').replace('.pq', ''),
        'num_clients': matrix.shape[0],
        'num_captains': matrix.shape[1],
        'matching_fn': matching_fn.__name__,
        'm': m,
        'capacity': capacity,
        'n_possible': n_possible,
        'dr': dr,
        'driver_frac': 1.0 - dropout,
        'num_clients_with_no_reach': idle_c,
        'num_captains_with_no_requests': idle_d,
        'mean_distance_to_client': m_dist,
        'num_clients_with_no_asks': idle_c_a,
        # 'avg_choice': a_stats['mean'],
        # 'median_choice': a_stats['median'],
        # 'q99_choice': a_stats['q_99'],
        'num_clients_with_no_handshake_options': idle_c_h
    }
