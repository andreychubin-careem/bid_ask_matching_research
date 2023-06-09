import sys
import numpy as np
from typing import Callable

sys.path.append('../')

from solver.matching import (
    greedy_sequential_matching,
    composite_sequential_matching,
    k_hungarian_matching,
    k_hungarian_capacity_exhaust,
    k_hungarian_m_capacity_exhaust
)
from solver.acceptance.captains import greedy_acceptance
from solver.acceptance.customers import greedy_handshake, imaginary_multicasting_handshake
from utils.functions import get_waiting, get_choice_stats, get_cost_matrix


def calculate(
        dist_matrix: np.ndarray,
        m: int,
        capacity: int,
        n_possible: int,
        dr: float = np.inf,
        drivers_dropout: float = 0.0,
        customers_dropout: float = 0.0,
        matching_fn: Callable = greedy_sequential_matching,
        handshake_fn: Callable = greedy_handshake,
        output: bool = False,
        verbose: bool = True
):
    n_cust, n_cap = dist_matrix.shape
    if verbose:
        print(f'Step 0. Num customers: {n_cust}, num captains: {n_cap}')
        print(' ')

    assignments = matching_fn(dist_matrix, m=m, capacity=capacity, dr=dr)
    waiting = get_waiting(assignments)
    if verbose:
        print(f'Step 1. Num customers with no reach: {waiting[0]}')
        print(f'Step 1. Num captains with no bids: {waiting[1]}')
        print(' ')

    acceptance = greedy_acceptance(dist_matrix, assignments, n_possible=n_possible, dropout=drivers_dropout)
    waiting = get_waiting(acceptance)
    if verbose:
        print(f'Step 2. Num customers with no asks: {waiting[0]}')
        print(f'Step 2. Num customers with exactly 1 ask: {len(np.where(acceptance.sum(axis=1) == 1.0)[0])}')
        print(f'Step 2. Num customers with exactly 2 asks: {len(np.where(acceptance.sum(axis=1) == 2.0)[0])}')
        print(f"Step 2. Customer's choice stats: {get_choice_stats(acceptance)}")
        print(' ')

    handshakes = handshake_fn(dist_matrix, acceptance, dropout=customers_dropout)
    waiting = get_waiting(handshakes)
    if verbose:
        print(f'Step 3. Num customers with no handshakes: {waiting[0]}')

    if output:
        return assignments, acceptance, handshakes


def main(matrix: np.ndarray, matching_fn: Callable, params: dict) -> None:
    print(' ')
    print(f'{matching_fn.__name__} approach:')
    calculate(
        dist_matrix=matrix,
        matching_fn=matching_fn,
        **params
    )
    _, acc, handshake = calculate(
        dist_matrix=matrix,
        matching_fn=matching_fn,
        handshake_fn=imaginary_multicasting_handshake,
        output=True,
        verbose=False,
        **params
    )
    print(' ')
    print(round(len(np.where(acc.sum(axis=0) > 1.0)[0]) / matrix.shape[1], 4))
    print(round(len(np.where(handshake.sum(axis=0) > 1.0)[0]) / matrix.shape[1], 4))
    print('-' * 50)


if __name__ == '__main__':
    np.random.seed(123)

    params = dict(
        m=7,
        capacity=5,
        n_possible=5,
        dr=15.0,
        drivers_dropout=0.0,
        customers_dropout=0.0
    )

    cost_matrix = get_cost_matrix(dropout=0.5)

    fns = [
        # random_closest_matching,
        # greedy_sequential_matching,
        composite_sequential_matching,
        k_hungarian_matching,
        # k_hungarian_capacity_exhaust,
        k_hungarian_m_capacity_exhaust,
        # k_hungarian_capacity_exhaust_random,
        # k_hungarian_m_capacity_exhaust_random
    ]

    for fn in fns:
        main(matrix=cost_matrix, matching_fn=fn, params=params)
