"""
tests for utils
"""

from utils import get_prediction_accuracy, get_df_r2score
import numpy as np
import pandas as pd


def test_accuracy():
    rng = np.random.default_rng(seed=1)
    true_maps = rng.normal(size=(2, 100))
    predicted_maps = true_maps + rng.normal(size=(2, 100)) * .1
    accuracy, _ = get_prediction_accuracy(predicted_maps, true_maps)
    assert accuracy


def test_accuracy_fail():
    rng = np.random.default_rng(seed=1)
    true_maps = rng.normal(size=(2, 100))
    predicted_maps = np.flipud(true_maps) + rng.normal(size=(2, 100)) * .1
    accuracy, _ = get_prediction_accuracy(predicted_maps, true_maps)
    assert not accuracy


def test_df_r2score():
    df1 = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4]}, index=['a', 'c', 'b', 'd'])
    scores = get_df_r2score(df1, df2)
    assert all(scores == 1)
