import numpy as np


def greedy_handshake(distance_matrix: np.ndarray, acceptance_matrix: np.ndarray, dropout: float = 0.0) -> np.ndarray:
    assert distance_matrix.shape == acceptance_matrix.shape, 'distance_matrix and acceptance_matrix shapes do not ' \
                                                             'match!'
    matrix = distance_matrix * acceptance_matrix
    handshakes = np.zeros_like(matrix)

    for i in range(matrix.shape[0]):
        asks = np.where(matrix[i] > 0.0)[0]

        if len(asks) == 0:
            continue

        if dropout > 0.0:
            num = len(asks)
            cols = np.sort(np.random.choice(np.arange(0, num), size=int(np.ceil(num * (1.0 - dropout))), replace=False))
            asks = asks[cols]

        index = np.argsort(matrix[i, asks], kind='quicksort')[0]
        ask_index = asks[index]
        handshakes[i, ask_index] = 1.0
        matrix[:, ask_index] *= 0.0  # make selected driver unavailable

    return handshakes


def improved_greedy_handshake(distance_matrix: np.ndarray, acceptance_matrix: np.ndarray) -> np.ndarray:
    # TODO: improve greedy_handshake by checking single alternative customers
    # Probably this approach makes no sense
    pass


def imaginary_multicasting_handshake(
        distance_matrix: np.ndarray,
        acceptance_matrix: np.ndarray,
        dropout: float = 0.0
) -> np.ndarray:
    assert distance_matrix.shape == acceptance_matrix.shape, 'distance_matrix and acceptance_matrix shapes do not ' \
                                                             'match!'
    matrix = distance_matrix * acceptance_matrix
    handshakes = np.zeros_like(matrix)

    for i in range(matrix.shape[0]):
        asks = np.where(matrix[i] > 0.0)[0]

        if len(asks) == 0:
            continue

        if dropout > 0.0:
            num = len(asks)
            cols = np.sort(np.random.choice(np.arange(0, num), size=int(np.ceil(num * (1.0 - dropout))), replace=False))
            asks = asks[cols]

        index = np.argsort(matrix[i, asks], kind='quicksort')[0]
        ask_index = asks[index]
        handshakes[i, ask_index] = 1.0
        # matrix[:, ask_index] *= 0.0  # make selected driver unavailable

    return handshakes

