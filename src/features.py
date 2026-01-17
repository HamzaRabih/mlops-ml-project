# src/features.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# (exemple: no-op pédagogique via FunctionTransformer)
from sklearn.preprocessing import FunctionTransformer

def _clip2(X):
    return X.clip(-3, 3)



def build_numeric_preprocess():
    """
    Prétraitement minimal (baseline) :
    - imputation médiane
    - clipping des valeurs extrêmes
    - standardisation
    """
    return Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ])