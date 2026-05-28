"""
統一 trainer：以 isinstance(model, nn.Module) 判斷 DL / sklearn 路徑。

ONNX 輸出統一使用 output_names=["label"]，deploy 端不需另外處理：
  - sklearn : label = 字串類別（skl2onnx 原生輸出）
  - DL      : label = int64 class index，搭配 label_classes.txt 查表
"""
from pathlib import Path
import numpy as np
import os
import sys

from sklearn.metrics import classification_report, f1_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ── Public API ─────────────────────────────────────────────────────────────────

def fit(model, X_train, y_train, label_encoder=None, train_params=None, device="auto"):
    if _TORCH_OK and isinstance(model, nn.Module):
        return _fit_dl(model, X_train, y_train, label_encoder, train_params or {}, device)
    return _fit_sklearn(model, X_train, y_train)


def evaluate(model, X_test, y_test, label_encoder=None):
    if _TORCH_OK and isinstance(model, nn.Module):
        return _evaluate_dl(model, X_test, y_test, label_encoder)
    return _evaluate_sklearn(model, X_test, y_test)


def export_onnx(model, feature_cols, output_path, label_encoder=None):
    if _TORCH_OK and isinstance(model, nn.Module):
        return _export_onnx_dl(model, feature_cols, output_path, label_encoder)
    return _export_onnx_sklearn(model, feature_cols, output_path)


# ── DL internals ───────────────────────────────────────────────────────────────

def _fit_dl(model, X_train, y_train, label_encoder, train_params, device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[train] device = {device}")
    model = model.to(device)

    y_int  = label_encoder.transform(y_train).astype(np.int64)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_int,   dtype=torch.long),
        ),
        batch_size=train_params.get("batch_size", 256),
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.get("lr", 1e-3))
    criterion = nn.CrossEntropyLoss()
    epochs    = train_params.get("epochs", 30)
    log_every = max(1, epochs // 5)

    print(f"[train] {model.__class__.__name__}  epochs={epochs}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_b)
        if epoch % log_every == 0:
            avg = total_loss / len(loader.dataset)
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    model = model.cpu()
    print("[train] done")
    return model


def _evaluate_dl(model, X_test, y_test, label_encoder):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        y_pred = label_encoder.inverse_transform(logits.argmax(dim=1).numpy())

    print("\n[evaluate] Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    return {
        "accuracy": round(float((y_pred == y_test).mean()), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
    }


class _ArgmaxWrapper(nn.Module):
    """logits → argmax，讓 ONNX 輸出 int64 class index（命名為 label）。"""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.backbone(x).argmax(dim=1)


def _export_onnx_dl(model, feature_cols, output_path, label_encoder):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _ArgmaxWrapper(model).eval()
    dummy   = torch.zeros(1, len(feature_cols))

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["float_input"],
        output_names=["label"],          # 與 sklearn ONNX 統一命名
        dynamic_axes={
            "float_input": {0: "batch_size"},
            "label":       {0: "batch_size"},
        },
        opset_version=17,
    )

    # deploy 端用 index 查表
    classes_path = output_path.parent / "label_classes.txt"
    classes_path.write_text("\n".join(label_encoder.classes_))
    print(f"[export] label_classes.txt → {classes_path}")
    print(f"[export] {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


# ── sklearn internals ──────────────────────────────────────────────────────────

def _fit_sklearn(model, X_train, y_train):
    print(f"\n[train] {model.__class__.__name__} fitting on {X_train.shape} ...")
    model.fit(X_train, y_train)
    print("[train] done")
    return model


def _evaluate_sklearn(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n[evaluate] Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    return {
        "accuracy": round(float((y_pred == y_test).mean()), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
    }


def _get_onnx_options(model):
    options = {}
    steps = model.steps if isinstance(model, Pipeline) else [(None, model)]
    for _, step in steps:
        if isinstance(step, LinearSVC):
            options[LinearSVC] = {"nocl": True}
    return options


def _export_onnx_sklearn(model, feature_cols, output_path):
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
        options    = _get_onnx_options(model)
        onnx_model = convert_sklearn(
            model, initial_types=initial_type, target_opset=17, options=options
        )
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