from sklearn.neighbors import KNeighborsClassifier
from onnx_utils import export_sklearn

NAME = "knn"

DEFAULT_PARAMS = {
    "n_neighbors": 5,
    "weights":     "uniform",
    "metric":      "minkowski",
    "n_jobs":      -1,
}


def build(params: dict) -> KNeighborsClassifier:
    return KNeighborsClassifier(**params)


def export_onnx(model, feature_cols, output_path, **kwargs):
    return export_sklearn(model, feature_cols, output_path)