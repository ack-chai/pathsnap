import unittest

import numpy as np
import pandas as pd

from pathsnap.hmm import viterbi


class TinyRoadNetwork:
    """Two directed 10 m arcs: 0 -> 1 -> 2."""

    def __init__(self):
        self.length = np.array([10.0, 10.0])
        self.src = np.array([0, 1])
        self.dst = np.array([1, 2])

    def dists_from(self, node, limit):
        if node == 0 and limit >= 10:
            return {0: 0.0, 1: 10.0}
        if node == 1 and limit >= 10:
            return {1: 0.0, 2: 10.0}
        return {node: 0.0}


class ViterbiTests(unittest.TestCase):
    def setUp(self):
        self.net = TinyRoadNetwork()
        self.params = {
            "beta": 5.0,
            "emission_power": 1.0,
            "gap_seconds": 180.0,
            "max_speed_mps": 50.0,
            "unreachable_logp": -30.0,
            "top_m": 8,
        }

    def test_decodes_connected_candidate_sequence(self):
        xy = np.array([[0.0, 0.0], [10.0, 0.0]])
        t = np.array([0.0, 1.0])
        frame = pd.DataFrame({
            "pidx": [0, 1], "arc": [0, 1], "t_along": [0.5, 0.5], "p": [0.9, 0.9]
        })

        np.testing.assert_array_equal(
            viterbi(self.net, xy, t, frame, self.params), np.array([0, 1])
        )

    def test_missing_candidate_breaks_sequence_and_marks_point_unmatched(self):
        xy = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        t = np.array([0.0, 1.0, 2.0])
        frame = pd.DataFrame({
            "pidx": [0, 2], "arc": [0, 1], "t_along": [0.5, 0.5], "p": [0.9, 0.8]
        })

        np.testing.assert_array_equal(
            viterbi(self.net, xy, t, frame, self.params), np.array([0, -1, 1])
        )

    def test_empty_candidate_table_returns_unmatched_points(self):
        xy = np.array([[0.0, 0.0], [1.0, 0.0]])
        t = np.array([0.0, 1.0])
        frame = pd.DataFrame(columns=["pidx", "arc", "t_along", "p"])

        np.testing.assert_array_equal(
            viterbi(self.net, xy, t, frame, self.params), np.array([-1, -1])
        )

    def test_rejects_mismatched_track_lengths(self):
        frame = pd.DataFrame({"pidx": [0], "arc": [0], "t_along": [0.5], "p": [0.9]})
        with self.assertRaisesRegex(ValueError, "same number of observations"):
            viterbi(self.net, np.zeros((2, 2)), np.array([0.0]), frame, self.params)

    def test_rejects_candidate_index_outside_track(self):
        frame = pd.DataFrame({"pidx": [2], "arc": [0], "t_along": [0.5], "p": [0.9]})
        with self.assertRaisesRegex(ValueError, "within the observation range"):
            viterbi(self.net, np.zeros((2, 2)), np.array([0.0, 1.0]), frame, self.params)

    def test_rejects_candidates_for_empty_track(self):
        frame = pd.DataFrame({"pidx": [0], "arc": [0], "t_along": [0.5], "p": [0.9]})
        with self.assertRaisesRegex(ValueError, "within the observation range"):
            viterbi(self.net, np.empty((0, 2)), np.array([]), frame, self.params)


if __name__ == "__main__":
    unittest.main()
