from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def fit(model: nn.Module, X_train, y_train, label_encoder, train_params: dict, device="auto") -> nn.Module:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[train] device = {device}")
    model = model.to(device)

    y_int = label_encoder.transform(y_train).astype(np.int64)
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


def evaluate(model: nn.Module, X_test, y_test, label_encoder) -> dict:
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


def export_onnx(model: nn.Module, feature_cols, output_path: Path, label_encoder) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, len(feature_cols)),
        str(output_path),
        input_names=["float_input"],
        output_names=["logits"],
        dynamic_axes={"float_input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )

    classes_path = output_path.parent / "label_classes.txt"
    classes_path.write_text("\n".join(label_encoder.classes_))
    print(f"[export] label_classes.txt → {classes_path}")
    print(f"[export] {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path