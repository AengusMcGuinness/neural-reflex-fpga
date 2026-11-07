import unittest

import numpy as np

from neural_reflex.pipeline import (fit_decoder, metrics, predict, split_windows,
                                    validate_emg, window_features)


class PipelineTests(unittest.TestCase):
    def test_window_features_skip_label_transitions(self):
        rows = [{"t_ns": index, "emg": [float(index)] * 8,
                 "label": 0 if index < 4 else 1, "round": 0}
                for index in range(8)]
        features, targets, _, starts = window_features(rows, 4, 2, labeled=True)
        self.assertEqual(starts, [0, 4])
        self.assertEqual(targets.tolist(), [0, 1])
        self.assertEqual(features.shape, (2, 16))
        self.assertAlmostEqual(features[0, 0], 1.0)
        self.assertAlmostEqual(features[0, 8], 1.0)

    def test_holdout_round_is_unseen_and_decoder_separates_signal(self):
        rng = np.random.default_rng(9)
        features = []
        targets = []
        rounds = []
        for round_id in range(3):
            for label in (0, 1):
                for _ in range(20):
                    features.append(rng.normal(label * 3, 0.2, 16))
                    targets.append(label)
                    rounds.append(round_id)
        features = np.asarray(features)
        targets = np.asarray(targets)
        train, test = split_windows(rounds, list(range(len(rounds))), 4)
        self.assertEqual({rounds[index] for index in np.where(test)[0]}, {2})
        model = fit_decoder(features, targets, train)
        self.assertGreater(metrics(targets[test], predict(model, features[test]))
                           ["accuracy_at_0_5"], 0.95)

    def test_rejects_non_raw_channel_shape(self):
        with self.assertRaises(ValueError):
            validate_emg([1, 2, 3])
        with self.assertRaises(ValueError):
            validate_emg([float("nan")] * 8)


if __name__ == "__main__":
    unittest.main()
