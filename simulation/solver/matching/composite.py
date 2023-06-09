import numpy as np
from scipy.optimize import linear_sum_assignment


def composite_sequential_matching(matrix: np.ndarray, m: int, capacity: int, dr: float = np.inf) -> np.ndarray:
    """
    Composite matching hungarian assignment + secondary assignment.
    :param matrix: distance customers-captains matrix
    :param m: number of captains that will get a single bid (customer's reach)
    :param capacity: number of bids that a single driver can receive
    :param dr: dispatch radius (km)
    :return: assignment matrix
    """
    assert np.isnan(matrix).astype(np.int8).sum() == 0, 'Cost matrix contains NaNs'
    assert np.isinf(np.abs(matrix)).astype(np.int8).sum() == 0, 'Cost matrix contains inf'

    sub = matrix.copy()

    # correction for common error in data
    sub[sub == 0.0] = 0.001

    # get primary assignments
    primary = np.zeros_like(sub)
    row, col = linear_sum_assignment(np.where(sub > dr, 999.0, sub))
    primary[row, col] = 1.0
    primary = np.where(sub > dr, 0.0, primary)

    if m == 1 and capacity == 1:
        return primary

    # initialization
    assignments = primary.astype(np.float32).copy()
    drivers_assignments = primary.sum(axis=0)
    capacity_constraints = np.ones((matrix.shape[1],)) * capacity
    captains = np.arange(start=0, stop=matrix.shape[1], step=1, dtype=np.int32)

    # reduce the capacity if captains assigned by hungarian
    capacity_constraints = np.where(drivers_assignments > 0.0, capacity_constraints - 1.0, capacity_constraints)

    if capacity == 1:
        not_available = np.where(capacity_constraints == 0.0)[0].copy()  # find drivers with no remaining capacity
        captains = np.delete(captains, not_available)
        capacity_constraints = np.delete(capacity_constraints, not_available)
        sub = np.delete(sub, not_available, axis=1)

    for i in range(assignments.shape[0]):
        if len(capacity_constraints) == 0:
            break

        indexes = np.argsort(sub[i], kind='quicksort')[:m]
        drivers = captains[indexes]  # select m closest drivers
        primary_captain = np.where(primary[i] == 1.0)[0]

        if len(primary_captain) > 0:
            if m == 1:
                # captain already assigned to customer by hungarian
                continue
            else:
                # remove driver that was already chosen by hungarian
                drivers = drivers[drivers != primary_captain[0]]

            if len(drivers) == m:
                # if driver chosen by hungarian was not in list, remove last driver
                indexes = indexes[:-1]
                drivers = captains[indexes]

        if not np.isinf(dr):
            close = np.where(sub[i] < dr)[0]  # select drivers within dispatch radius
            indexes = indexes[np.isin(indexes, close)]  # exclude drivers beyond dispatch radius
            drivers = captains[indexes]

        if len(drivers) > 0:
            assignments[i, drivers] = 1.0  # set assignment flag

            capacity_constraints[indexes] -= 1.0  # reduce drivers' capacity
            not_available = np.where(capacity_constraints == 0.0)[0].copy()  # find drivers with no remaining capacity
            captains = np.delete(captains, not_available)
            capacity_constraints = np.delete(capacity_constraints, not_available)
            sub = np.delete(sub, not_available, axis=1)

    return assignments
