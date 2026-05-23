from pathlib import Path

import numpy as np

from sklearn.metrics import classification_report, f1_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import os, sys

def _get_onnx_options(model):
    options = {}
    steps = model.steps if isinstance(model, Pipeline) else [(None, model)]
    for _, step in steps:
        if isinstance(step, LinearSVC):
            options[LinearSVC] = {"nocl": True}
    return options

def fit(model, X_train, y_train):
    print(f"\n[train] {model.__class__.__name__} fitting on {X_train.shape} ...")
    model.fit(X_train, y_train)
    print("[train] done")
    return model


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    print("\n[evaluate] Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    return {
        "accuracy": round(float((y_pred == y_test).mean()), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
    }


import os, sys

def export_onnx(model, feature_cols, output_path) -> Path:
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
        options = _get_onnx_options(model)
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17, options=options)
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