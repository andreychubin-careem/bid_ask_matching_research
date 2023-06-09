import numpy as np
from scipy.optimize import linear_sum_assignment


def k_hungarian_matching(matrix: np.ndarray, m: int, capacity: int, dr: float = np.inf) -> np.ndarray:
    """
    k-hungarian hungarian assignment.
    :param matrix: distance customers-captains matrix
    :param m: number of captains that will get a single bid (customer's reach)
    :param capacity: number of bids that a single driver can receive
    :param dr: dispatch radius (km)
    :return: assignment matrix
    """
    assert np.isnan(matrix).astype(np.int8).sum() == 0, 'Cost matrix contains NaNs'
    assert np.isinf(np.abs(matrix)).astype(np.int8).sum() == 0, 'Cost matrix contains inf'

    sub = matrix.copy()

    # correction for common errors in data
    sub[sub == 0.0] = 0.001

    # initialization
    assignments = np.zeros_like(sub)
    capacity_constraints = np.ones((sub.shape[1],)) * capacity
    captains = np.arange(start=0, stop=matrix.shape[1], step=1, dtype=np.int32)

    # while np.where(np.isinf(capacity_constraints), 0.0, capacity_constraints).sum() > 0.0:
    for _ in range(m):
        row, col = linear_sum_assignment(np.where(sub > dr, 999.0, sub))
        capacity_constraints[col] -= 1.0  # reduce drivers' capacity
        assignments[row, captains[col]] = 1.0
        sub[row, col] = 9999.0
        not_available = np.where(capacity_constraints == 0.0)[0].copy()
        captains = np.delete(captains, not_available)
        capacity_constraints = np.delete(capacity_constraints, not_available)
        sub = np.delete(sub, not_available, axis=1)

    assignments = np.where(matrix > dr, 0.0, assignments)

    return assignments


def k_hungarian_capacity_exhaust(matrix: np.ndarray, m: int, capacity: int, dr: float = np.inf) -> np.ndarray:
    """
    k-hungarian hungarian assignment.
    :param matrix: distance customers-captains matrix
    :param m: number of captains that will get a single bid (customer's reach) (compatibility only)
    :param capacity: number of bids that a single driver can receive
    :param dr: dispatch radius (km)
    :return: assignment matrix
    """
    assert np.isnan(matrix).astype(np.int8).sum() == 0, 'Cost matrix contains NaNs'
    assert np.isinf(np.abs(matrix)).astype(np.int8).sum() == 0, 'Cost matrix contains inf'

    sub = matrix.copy()

    # correction for common errors in data
    sub[sub == 0.0] = 0.001

    # initialization
    assignments = np.zeros_like(sub)
    capacity_constraints = np.ones((sub.shape[1],)) * capacity
    captains = np.arange(start=0, stop=matrix.shape[1], step=1, dtype=np.int32)

    while len(capacity_constraints) > 0:
        row, col = linear_sum_assignment(np.where(sub > dr, 999.0, sub))
        capacity_constraints[col] -= 1.0  # reduce drivers' capacity
        assignments[row, captains[col]] = 1.0
        sub[row, col] = 9999.0
        not_available = np.where(capacity_constraints == 0.0)[0].copy()
        captains = np.delete(captains, not_available)
        capacity_constraints = np.delete(capacity_constraints, not_available)
        sub = np.delete(sub, not_available, axis=1)

    assignments = np.where(matrix > dr, 0.0, assignments)

    return assignments


def k_hungarian_m_capacity_exhaust(matrix: np.ndarray, m: int, capacity: int, dr: float = np.inf) -> np.ndarray:
    """
    k-hungarian hungarian assignment.
    :param matrix: distance customers-captains matrix
    :param m: number of captains that will get a single bid (customer's reach) (compatibility only)
    :param capacity: number of bids that a single driver can receive
    :param dr: dispatch radius (km)
    :return: assignment matrix
    """
    assert np.isnan(matrix).astype(np.int8).sum() == 0, 'Cost matrix contains NaNs'
    assert np.isinf(np.abs(matrix)).astype(np.int8).sum() == 0, 'Cost matrix contains inf'

    sub = matrix.copy()

    # correction for common errors in data
    sub[sub == 0.0] = 0.001

    # initialization
    assignments = np.zeros_like(sub)
    captains_capacity_constraints = np.ones((sub.shape[1],)) * capacity
    customers_capacity_constraints = np.ones((sub.shape[0],)) * m
    captains = np.arange(start=0, stop=matrix.shape[1], step=1, dtype=np.int32)
    customers = np.arange(start=0, stop=matrix.shape[0], step=1, dtype=np.int32)

    while len(captains_capacity_constraints) > 0 and len(customers_capacity_constraints) > 0:
        row, col = linear_sum_assignment(np.where(sub > dr, 999.0, sub))
        assignments[customers[row], captains[col]] = 1.0
        sub[row, col] = 9999.0

        captains_capacity_constraints[col] -= 1.0  # reduce drivers' capacity
        not_available = np.where(captains_capacity_constraints == 0.0)[0].copy()
        captains = np.delete(captains, not_available)
        captains_capacity_constraints = np.delete(captains_capacity_constraints, not_available)
        sub = np.delete(sub, not_available, axis=1)

        customers_capacity_constraints[row] -= 1.0  # reduce bids capacity
        filled_bids = np.where(customers_capacity_constraints == 0.0)[0].copy()
        customers = np.delete(customers, filled_bids)
        customers_capacity_constraints = np.delete(customers_capacity_constraints, filled_bids)
        sub = np.delete(sub, filled_bids, axis=0)

    assignments = np.where(matrix > dr, 0.0, assignments)

    return assignments
