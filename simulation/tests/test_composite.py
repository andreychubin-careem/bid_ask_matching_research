import sys
import unittest
import numpy as np
import pandas as pd

sys.path.append('../')

from solver.matching import composite_sequential_matching
from solver.acceptance.captains import greedy_acceptance
from solver.acceptance.customers import greedy_handshake
from utils.functions import get_waiting


class CompositeSolverTestCase(unittest.TestCase):
    def make_matrix(self, m: int = 10, capacity: int = 10, dr: float = 10.0):
        data = pd.read_parquet('_data/sample_snapshot_2.parquet')
        self.dist_matrix = pd.pivot_table(data, values='distance', index='userid', columns='driver_id').values
        self.assignments = composite_sequential_matching(self.dist_matrix, m=m, capacity=capacity, dr=dr)
        self.acceptance = greedy_acceptance(self.dist_matrix, self.assignments, n_possible=2, dropout=0.1)
        self.handshakes = greedy_handshake(self.dist_matrix, self.acceptance)
        self.matrix_exist = True

    def check_data(self):
        try:
            if not self.matrix_exist:
                self.make_matrix()
        except AttributeError:
            self.make_matrix()

    def test_dist_matrix_nans(self):
        self.check_data()
        n_nans = np.isnan(self.dist_matrix).astype(np.int8).sum()
        self.assertEqual(n_nans, 0.0)

    def test_dist_matrix_infs(self):
        self.check_data()
        n_infs = np.isinf(np.abs(self.dist_matrix)).astype(np.int8).sum()
        self.assertEqual(n_infs, 0.0)

    def test_dist_matrix_neg(self):
        self.check_data()
        n_neg = len(np.where(self.dist_matrix < 0.0)[0])
        self.assertEqual(n_neg, 0.0)

    def test_assignments_shape(self):
        self.check_data()
        self.assertEqual(self.dist_matrix.shape, self.assignments.shape)

    def test_assignments_nans(self):
        self.check_data()
        n_nans = np.isnan(self.assignments).astype(np.int8).sum()
        self.assertEqual(n_nans, 0.0)

    def test_assignments_infs(self):
        self.check_data()
        n_infs = np.isinf(np.abs(self.assignments)).astype(np.int8).sum()
        self.assertEqual(n_infs, 0.0)

    def test_assignments_values(self):
        self.check_data()
        unique = np.unique(self.assignments)
        self.assertIn(len(unique), [1, 2])
        self.assertIn(unique[0], [0.0, 1.0])
        self.assertIn(unique[1], [0.0, 1.0])

    def test_assignments_n_zero_case(self):
        self.make_matrix(m=1, capacity=1, dr=np.inf)
        self.matrix_exist = False
        cust, cap = get_waiting(self.assignments)
        self.assertAlmostEqual(
            max(cust, cap),
            np.abs(self.assignments.shape[1] - self.assignments.shape[0]),
            delta=min(self.assignments.shape[0], self.assignments.shape[1]) * 0.05  # TODO: replace insecure delta
        )

    def test_acceptance_shape(self):
        self.check_data()
        self.assertEqual(self.dist_matrix.shape, self.acceptance.shape)

    def test_acceptance_nans(self):
        self.check_data()
        n_nans = np.isnan(self.acceptance).astype(np.int8).sum()
        self.assertEqual(n_nans, 0.0)

    def test_acceptance_infs(self):
        self.check_data()
        n_infs = np.isinf(np.abs(self.acceptance)).astype(np.int8).sum()
        self.assertEqual(n_infs, 0.0)

    def test_acceptance_values(self):
        self.check_data()
        unique = np.unique(self.acceptance)
        self.assertIn(len(unique), [1, 2])
        self.assertIn(unique[0], [0.0, 1.0])
        self.assertIn(unique[1], [0.0, 1.0])

    def test_handshake_shapes(self):
        self.check_data()
        self.assertEqual(self.dist_matrix.shape, self.handshakes.shape)

    def test_handshake_consistency(self):
        self.check_data()
        self.assertLessEqual(self.handshakes.sum(), min(self.handshakes.shape[0], self.handshakes.shape[1]))

    def test_handshake_nans(self):
        self.check_data()
        n_nans = np.isnan(self.handshakes).astype(np.int8).sum()
        self.assertEqual(n_nans, 0.0)

    def test_handshake_infs(self):
        self.check_data()
        n_infs = np.isinf(np.abs(self.handshakes)).astype(np.int8).sum()
        self.assertEqual(n_infs, 0.0)

    def test_handshake_values(self):
        self.check_data()
        unique = np.unique(self.handshakes)
        self.assertIn(len(unique), [1, 2])
        self.assertIn(unique[0], [0.0, 1.0])
        self.assertIn(unique[1], [0.0, 1.0])


if __name__ == '__main__':
    unittest.main()
