import numpy as np


def _match_closest_driver(matrix: np.ndarray, m: int) -> np.ndarray:
    indices = np.argsort(matrix, kind='quicksort')[:, :m]
    mask = np.zeros_like(matrix)

    for i in range(mask.shape[0]):
        mask[i][indices[i]] = 1.0

    return mask


def _check_capacity(matrix: np.ndarray, capacity: int) -> np.ndarray:
    sub = matrix.copy()
    current = sub.sum(axis=0)
    indices = np.where(current > capacity)[0]

    for i in indices:
        curr = np.where(sub[:, i] == 1.0)[0]
        to_drop_ind = np.random.choice(curr, size=np.int32(len(curr) - capacity), replace=False)
        sub[:, i][to_drop_ind] = 0.0

    return sub


def random_closest_matching(matrix: np.ndarray, m: int, capacity: int, dr: float = np.inf) -> np.ndarray:
    """
    Inaccurate but fast matching algorithm. Deprecated.
    :param matrix: distance customers-captains matrix
    :param m: number of captains that will get a single bid (customer's reach)
    :param capacity: number of bids that a single driver can receive
    :param dr: dispatch radius (km)
    :return: assignment matrix
    """
    if np.isinf(dr):
        return _check_capacity(_match_closest_driver(matrix, m), capacity)

    else:
        sub = matrix.copy()
        sub[sub > dr] = np.inf
        mask = _match_closest_driver(sub, m)
        mask *= sub
        mask[np.isnan(mask)] = 0.0
        mask[np.isinf(mask)] = 0.0
        mask[mask > 0.0] = 1.0

        return _check_capacity(mask, capacity)
