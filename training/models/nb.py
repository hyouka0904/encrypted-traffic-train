from sklearn.naive_bayes import GaussianNB
from onnx_utils import export_sklearn

NAME = "nb"

DEFAULT_PARAMS = {
    "var_smoothing": 1e-9,
}


def build(params: dict) -> GaussianNB:
    return GaussianNB(**params)


def export_onnx(model, feature_cols, output_path, **kwargs):
    return export_sklearn(model, feature_cols, output_path)