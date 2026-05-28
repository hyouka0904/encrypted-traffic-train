from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from onnx_utils import export_sklearn

NAME = "svm"

DEFAULT_PARAMS = {
    "C":        1.0,
    "max_iter": 2000,
}


def build(params: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LinearSVC(**params)),
    ])


def export_onnx(model, feature_cols, output_path, **kwargs):
    return export_sklearn(
        model, feature_cols, output_path,
        options={LinearSVC: {"nocl": True}},
    )