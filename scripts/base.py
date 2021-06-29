from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np

class Encode(BaseEstimator, TransformerMixin):

    """encodes a dataframe based on list of features given"""

    def __init__(self,columns: list) -> None:
        super().__init__()
        self.label = LabelEncoder()
        self.columns = columns

    def fit(self):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X = X.copy()
        for column in X.columns:
            if column in self.columns:
                X[column] = self.label.fit_transform(X[column])
            else:
                continue

        return X


class SelectFeatures:

    """selects important features needed for prediction"""

    def __init__(self, features: list) -> None:
        self.features = features

    def transform(self,X):
        result = []
        for column in X.columns:
            if column in self.features:
                result.append(np.array(X[column]))
        return np.array(result)