from __future__ import annotations

import numpy as np

from gbm_ai.safety.ood_detection import (
    cosine_knn_distance,
    fit_ood_reference,
    mahalanobis_distance,
)


def test_ood_reference_and_distances_are_finite():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(40, 16))
    query = rng.normal(size=(5, 16))

    reference = fit_ood_reference(train, pca_components=8)
    distances = mahalanobis_distance(reference, query)

    assert distances.shape == (5,)
    assert np.isfinite(distances).all()
    assert np.all(distances >= 0.0)
    assert reference["pca_components"] == 8


def test_cosine_knn_distance_is_smaller_for_nearby_points():
    train = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    query = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    distances = cosine_knn_distance(train, query, k=2)
    assert distances[0] < distances[1]


def test_fit_reference_caps_pca_components_to_available_rank():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(10, 50))
    reference = fit_ood_reference(train, pca_components=32)
    assert reference["pca_components"] == 9
