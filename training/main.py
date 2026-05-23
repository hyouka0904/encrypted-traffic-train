import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))


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
    parser.add_argument(
    "--device",
    default="auto",
    choices=["cpu", "cuda", "auto"],
    help="training device (default: auto)",
)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name   = cfg["model"]["name"]
    model_module = importlib.import_module(f"models.{model_name}")
    is_dl        = getattr(model_module, "IS_DL", False)

    # 根據路線 import 對應 trainer
    trainer = importlib.import_module("train_dl" if is_dl else "train_sklearn")

    # model params：config 蓋過 DEFAULT_PARAMS
    model_params = {**model_module.DEFAULT_PARAMS, **cfg["model"].get("params", {})}

    output_dir = Path(cfg["output"]["dir"])
    onnx_path  = output_dir / f"{model_name}.onnx"

    print(f"\n{'='*50}")
    print(f"  model  : {model_name}  ({'DL' if is_dl else 'sklearn'})")
    print(f"  params : {model_params}")
    print(f"  output : {onnx_path}")
    print(f"{'='*50}")

    X_train, y_train, X_test, y_test, feature_cols = load_data(cfg["data"]["processed_dir"])

    if is_dl:
        # DL 專屬：LabelEncoder + train_params
        label_encoder = LabelEncoder().fit(y_train)
        n_classes     = len(label_encoder.classes_)

        # train_params：TRAIN_PARAMS 為 DL 訓練超參（epochs/lr/batch），可被 yaml 蓋過
        train_params = {
            **getattr(model_module, "TRAIN_PARAMS", {}),
            **cfg["model"].get("train_params", {}),
        }

        # DL 的 build 需要 n_features / n_classes
        model   = model_module.build(model_params, n_features=X_train.shape[1], n_classes=n_classes)
        model   = trainer.fit(model, X_train, y_train, label_encoder, train_params, device=args.device)
        metrics = trainer.evaluate(model, X_test, y_test, label_encoder)
        trainer.export_onnx(model, feature_cols, onnx_path, label_encoder)
    else:
        model   = model_module.build(model_params)
        model   = trainer.fit(model, X_train, y_train)
        metrics = trainer.evaluate(model, X_test, y_test)
        trainer.export_onnx(model, feature_cols, onnx_path)

    # features.txt 複製到 output（deploy 端需要）
    feat_src = Path(cfg["data"]["processed_dir"]) / "features.txt"
    if feat_src.exists():
        shutil.copy(feat_src, output_dir / "features.txt")

    # 結果存 JSON
    results = {
        "model": model_name,
        "type": "dl" if is_dl else "sklearn",
        "params": model_params,
        **metrics,
        "model_size_kb": round(onnx_path.stat().st_size / 1024, 1),
    }
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