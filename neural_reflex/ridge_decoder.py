"""Multi-output ridge regression from lagged EMG features to joint angles."""

import numpy as np


def split_by_block(block_ids, test_blocks, validation_blocks):
    blocks = np.asarray(list(block_ids))
    test = np.isin(blocks, list(test_blocks))
    validation = np.isin(blocks, list(validation_blocks))
    train = ~test & ~validation
    if not train.any() or not test.any():
        raise ValueError("train and test each need at least one block")
    return train, validation, test


def fit_ridge(features, targets, train_mask, regularization):
    train_x = np.asarray(features, dtype=np.float64)[train_mask]
    train_y = np.asarray(targets, dtype=np.float64)[train_mask]
    if train_y.ndim == 1:
        train_y = train_y[:, None]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (train_x - mean) / scale
    augmented = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.eye(augmented.shape[1]) * regularization
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ train_y)
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "bias": weights[0].tolist(), "weights": weights[1:].tolist(),
            "regularization": float(regularization)}


def predict_angles(model, features):
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = np.asarray(model["bias"], dtype=np.float64)
    return (np.asarray(features, dtype=np.float64) - mean) / scale @ weights + bias
