from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "svm"
DEFAULT_PARAMS = {
    "C": 1.0,
    "max_iter": 2000,
}

def build(params: dict):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(**params)),
    ])