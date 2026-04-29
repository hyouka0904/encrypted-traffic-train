import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


DATA_DIR  = Path("data/processed")
OUT_DIR   = Path("training")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = (DATA_DIR / "features.txt").read_text().splitlines()
    label_col = [c for c in train_df.columns if c.startswith("class")][0]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df[label_col].values

    print(f"[load] train: {X_train.shape}, test: {X_test.shape}")
    print(f"[load] features: {feature_cols}")
    return X_train, y_train, X_test, y_test, feature_cols


def train(X_train, y_train):
    print("\n[train] Random Forest ...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",  
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("[train] done")
    return clf


def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("\n[evaluate] Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Feature importance
    feature_cols = (DATA_DIR / "features.txt").read_text().splitlines()
    importances = sorted(
        zip(feature_cols, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("[evaluate] Top-10 Feature Importance:")
    for name, score in importances[:10]:
        print(f"  {name:<25} {score:.4f}")


def export_onnx(clf, feature_cols):
    n_features = len(feature_cols)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        target_opset=17,
    )

    onnx_path = OUT_DIR / "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\n[export] ONNX to {onnx_path}")
    print(f"[export] : {onnx_path.stat().st_size / 1024:.1f} KB")


def main():
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    clf = train(X_train, y_train)
    evaluate(clf, X_test, y_test)
    export_onnx(clf, feature_cols)


if __name__ == "__main__":
    main()