import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from train import evaluate, export_onnx, fit


def load_data(processed_dir: str) -> tuple:
    d = Path(processed_dir)
    train_df     = pd.read_csv(d / "train.csv")
    test_df      = pd.read_csv(d / "test.csv")
    feature_cols = (d / "features.txt").read_text().splitlines()
    label_col    = next(c for c in train_df.columns if c.startswith("class"))

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df[label_col].values

    print(f"[data] train={X_train.shape}  test={X_test.shape}")
    return X_train, y_train, X_test, y_test, feature_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    # config 裡的 params 蓋過 model 檔的 DEFAULT_PARAMS
    model_module = importlib.import_module(f"models.{model_name}")
    params = {**model_module.DEFAULT_PARAMS, **cfg["model"].get("params", {})}

    output_dir = Path(cfg["output"]["dir"])
    onnx_path  = output_dir / f"{model_name}.onnx"

    print(f"\n{'='*50}")
    print(f"  model  : {model_name}")
    print(f"  params : {params}")
    print(f"  output : {onnx_path}")
    print(f"{'='*50}")

    X_train, y_train, X_test, y_test, feature_cols = load_data(cfg["data"]["processed_dir"])

    model   = model_module.build(params)
    model   = fit(model, X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    export_onnx(model, feature_cols, onnx_path)

    # features.txt 也複製到 output（deploy 端需要）
    feat_src = Path(cfg["data"]["processed_dir"]) / "features.txt"
    if feat_src.exists():
        shutil.copy(feat_src, output_dir / "features.txt")

    # 結果存 JSON
    results = {"model": model_name, "params": params, **metrics,
               "model_size_kb": round(onnx_path.stat().st_size / 1024, 1)}
    results_path = output_dir / f"{model_name}_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*50}")
    print(f"  macro F1   : {metrics['macro_f1']:.4f}")
    print(f"  accuracy   : {metrics['accuracy']:.4f}")
    print(f"  model size : {results['model_size_kb']:.1f} KB")
    print(f"{'='*50}")
    print(f"\n[done] {results_path}")


if __name__ == "__main__":
    main()