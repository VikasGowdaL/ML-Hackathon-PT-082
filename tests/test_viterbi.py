import unittest
import numpy as np
import pandas as pd

from pathsnap.hmm.viterbi import ViterbiConfig, decode_viterbi


class TestViterbi(unittest.TestCase):
    def setUp(self):
        self.arcs = pd.DataFrame({
            "arc_id": [10, 11, 12, 20, 21],
            "from_node_id": [0, 1, 2, 8, 9],
            "to_node_id": [1, 2, 3, 9, 10],
        })

    def test_connected_path_wins_over_independent_argmax(self):
        c = pd.DataFrame({
            "obs_idx": [0, 0, 1, 1, 2, 2],
            "arc_id": [10, 20, 11, 21, 12, 20],
            "probability": [.9, .4, .55, .9, .9, .7],
        })
        out = decode_viterbi(c, self.arcs)
        self.assertEqual(out.predicted_arc_id.tolist(), [10, 11, 12])

    def test_missing_observation_breaks_chain(self):
        c = pd.DataFrame({
            "obs_idx": [0, 2], "arc_id": [10, 21], "probability": [.9, .9],
        })
        out = decode_viterbi(c, self.arcs, n_observations=3)
        self.assertEqual(out.status.tolist(), ["matched", "no_candidates", "matched"])
        self.assertTrue(pd.isna(out.loc[1, "predicted_arc_id"]))

    def test_reset_allows_disconnected_switch(self):
        c = pd.DataFrame({
            "obs_idx": [0, 1, 1], "arc_id": [10, 11, 21], "probability": [.95, .4, .95],
        })
        out = decode_viterbi(c, self.arcs, reset_at=np.array([False, True]))
        self.assertEqual(out.predicted_arc_id.tolist(), [10, 21])

    def test_invalid_probabilities_are_rejected(self):
        c = pd.DataFrame({"obs_idx": [0], "arc_id": [10], "probability": [1.5]})
        with self.assertRaises(ValueError):
            decode_viterbi(c, self.arcs)


if __name__ == "__main__":
    unittest.main()
