# src/features.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, min_val=-5, max_val=5):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(X, self.min_val, self.max_val)


def build_numeric_preprocess():
    """
    Prétraitement minimal (baseline) :
    - imputation médiane
    - clipping des valeurs extrêmes
    - standardisation
    """
    return Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    #("clipper", Clipper(-5, 5)),
    ("scaler", StandardScaler()),
    ])