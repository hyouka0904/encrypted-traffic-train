from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, f1_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import os, sys


def fit(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    print(f"\n[train] {model.__class__.__name__} fitting on {X_train.shape} ...")
    model.fit(X_train, y_train)
    print("[train] done")
    return model


def evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    print("\n[evaluate] Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    return {
        "accuracy": round(float((y_pred == y_test).mean()), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
    }


import os, sys

def export_onnx(model, feature_cols, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]

    sys.stdout.flush()
    sys.stderr.flush()
    old_out, old_err = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        os.close(devnull)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"\n[export] {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path