"""
encoding model class
using sklearn interface
"""

from sklearn.linear_model import LinearRegression, RidgeCV


class EncodingModel:
    def __init__(self, method='ridgecv', cv=None):
        self.method = method
        self.cv = cv  # default uses efficient gcv
        self.clf = None

    def fit(self, X, y):
        if self.method == 'lr':
            self.clf = LinearRegression()
        elif self.method == 'ridgecv':
            self.clf = RidgeCV(cv=self.cv)
        else:
            raise Exception(f'method {self.method} not implemented')

        self.clf.fit(X, y)

    def predict(self, X):
        return(self.clf.predict(X))
    
