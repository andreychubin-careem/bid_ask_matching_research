import numpy as np


def greedy_sequential_matching(matrix: np.ndarray, m: int, capacity: int, dr: float = np.inf) -> np.ndarray:
    """
    Greedy matching. Customers are processed sequentially (FIFO)
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

    for i in range(assignments.shape[0]):
        drivers = np.argsort(sub[i], kind='quicksort')[:m]  # select m closest drivers

        if np.isinf(dr):
            # for cases when dr == np.inf we need to explicitly exclude drivers with no capacity
            # (i.e. np.inf distance to client)
            inf_dist = np.where(np.isinf(sub[i]))[0]
            drivers = drivers[~np.isin(drivers, inf_dist)]
        else:
            close = np.where(sub[i] < dr)[0]  # select drivers within dispatch radius
            drivers = drivers[np.isin(drivers, close)]  # exclude drivers beyond dispatch radius

        assignments[i, drivers] = 1.0  # set assignment flag

        capacity_constraints[drivers] -= 1.0  # reduce drivers' capacity
        capacity_constraints[capacity_constraints == 0.0] = np.inf  # mark drivers with no capacity
        constraint = capacity_constraints.copy()
        constraint[~np.isinf(constraint)] = 1.0  # mark drivers with remaining capacity
        sub *= constraint  # set infinite distance to drivers with no capacity

    return assignments
