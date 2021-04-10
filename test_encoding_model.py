"""
encoding model tests
"""

import pytest
import numpy as np
from encoding_model import EncodingModel


@pytest.fixture
def encoding_model_lr():
    return(EncodingModel(method='lr'))


@pytest.fixture
def encoding_model_ridge():
    return(EncodingModel(method='ridgecv'))


@pytest.fixture
def testdata():
    rng = np.random.default_rng(seed=1)
    nrows = 100
    ncols = 20
    noise_sd = 0
    X = rng.normal(size=(nrows, ncols))
    beta = rng.random(ncols)
    noise = rng.normal(size=nrows, scale=noise_sd) if noise_sd > 0 else 0
    y = X.dot(beta) + noise
    return({'X': X, 'y': y, 'beta': beta})


# test  linear regression methods
def test_class_lr(encoding_model_lr):
    assert encoding_model_lr is not None


def test_fit_lr(encoding_model_lr, testdata):
    encoding_model_lr.fit(testdata['X'], testdata['y'])


# since we have not added any noise here, predicted should be very close to true
# use atol of 1e-2 to allow some small differences (should be guaranteed by using fixed seed)
def test_predict_lr(encoding_model_lr, testdata):
    encoding_model_lr.fit(testdata['X'], testdata['y'])
    predicted = encoding_model_lr.predict(testdata['X'])
    assert predicted.shape == testdata['y'].shape
    assert np.allclose(predicted, testdata['y'])
    assert(np.allclose(encoding_model_lr.clf.coef_, testdata['beta']))


# test ridge methods
def test_class_ridge(encoding_model_ridge):
    assert encoding_model_ridge is not None


def test_fit_ridge(encoding_model_ridge, testdata):
    encoding_model_ridge.fit(testdata['X'], testdata['y'])


# since we have not added any noise here, predicted should be very close to true
# use atol of 1e-2 to allow some small differences (should be guaranteed by using fixed seed)
def test_predict_ridge(encoding_model_ridge, testdata):
    encoding_model_ridge.fit(testdata['X'], testdata['y'])
    predicted = encoding_model_ridge.predict(testdata['X'])
    assert predicted.shape == testdata['y'].shape
    assert np.allclose(predicted, testdata['y'], atol=1e-2)
