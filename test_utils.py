"""
tests for utils
"""


from  utils import get_prediction_accuracy
import numpy as np


def test_accuracy():
    rng = np.random.default_rng(seed=1)
    true_maps = rng.normal(size=(2, 100))
    predicted_maps = true_maps + rng.normal(size=(2, 100))*.1
    accuracy = get_prediction_accuracy(predicted_maps, true_maps)
    assert accuracy

def test_accuracy_fail():
    rng = np.random.default_rng(seed=1)
    true_maps = rng.normal(size=(2, 100))
    predicted_maps = np.flipud(true_maps) + rng.normal(size=(2, 100))*.1
    accuracy = get_prediction_accuracy(predicted_maps, true_maps)
    assert not accuracy
