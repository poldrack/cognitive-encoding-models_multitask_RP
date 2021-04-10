"""
encoding model class
using sklearn interface
"""

from sklearn.linear_model import LinearRegression, RidgeCV
import numpy as np


class EncodingModel:
    def __init__(self, method='ridgecv', cv=None, alphas=None):
        self.method = method
        self.cv = cv  # default uses efficient gcv
        self.clf = None
        if alphas is not None:
            self.alphas = alphas
        else:
            self.alphas = [0.001, 0.01, 0.1]
            self.alphas.extend(np.linspace(1, 10, 10))

    def fit(self, X, y):
        if self.method == 'lr':
            self.clf = LinearRegression()
        elif self.method == 'ridgecv':
            self.clf = RidgeCV(cv=self.cv, alphas=self.alphas)
        else:
            raise Exception(f'method {self.method} not implemented')

        self.clf.fit(X, y)

    def predict(self, X):
        return(self.clf.predict(X))
    
