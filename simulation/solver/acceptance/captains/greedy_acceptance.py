import sys
import numpy as np


def greedy_acceptance(
        distance_matrix: np.ndarray,
        assignment_matrix: np.ndarray,
        n_possible: int,
        dropout: float
) -> np.ndarray:
    assert distance_matrix.shape == assignment_matrix.shape, 'distance_matrix and assignment_matrix shapes do not ' \
                                                             'match!'
    matrix = assignment_matrix * distance_matrix
    matrix = matrix.T
    acceptance = np.zeros_like(matrix)

    for i in range(matrix.shape[0]):
        # TODO: improve performance
        customers = np.where(matrix[i] != 0.0)[0]

        if len(customers) == 0:
            continue

        if dropout > 0.0:
            num = len(customers)
            cols = np.sort(np.random.choice(np.arange(0, num), size=int(np.ceil(num * (1.0-dropout))), replace=False))
            customers = customers[cols]

        indexes = np.argsort(matrix[i, customers], kind='quicksort')[:n_possible]
        selected = customers[indexes]
        acceptance[i, selected] = 1.0

    return acceptance.T
