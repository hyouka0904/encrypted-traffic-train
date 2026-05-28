from sklearn.ensemble import RandomForestClassifier
from onnx_utils import export_sklearn

NAME = "rf"

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth":    None,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs":       -1,
}


def build(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


def export_onnx(model, feature_cols, output_path, **kwargs):
    return export_sklearn(model, feature_cols, output_path)