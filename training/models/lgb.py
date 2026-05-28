from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

NAME = "lgb"

DEFAULT_PARAMS = {
    "n_estimators":  300,
    "max_depth":     6,
    "learning_rate": 0.1,
    "subsample":     0.8,
    "colsample_bytree": 0.8,
    "device":        "cpu",    # 沒 GPU 改成 "cpu"
    "n_jobs":        -1,
}


class LGBWrapper:
    """在模型內部處理 label encoding，讓外部統一用字串 label。"""

    def __init__(self, lgb_model):
        self.model = lgb_model
        self.le    = LabelEncoder()

    def fit(self, X, y):
        self.model.fit(X, self.le.fit_transform(y))
        return self

    def predict(self, X):
        return self.le.inverse_transform(self.model.predict(X))

    @property
    def classes_(self):
        return self.le.classes_


def build(params: dict) -> LGBWrapper:
    return LGBWrapper(LGBMClassifier(**params, random_state=42, verbose=-1))


def export_onnx(model, feature_cols, output_path, **kwargs):
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    onnx_model = convert_lightgbm(
        model.model,
        initial_types=[("float_input", FloatTensorType([None, len(feature_cols)]))],
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    classes_path = output_path.parent / "label_classes.txt"
    classes_path.write_text("\n".join(str(c) for c in model.classes_))
    print(f"[export] label_classes.txt → {classes_path}")
    print(f"[export] {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path