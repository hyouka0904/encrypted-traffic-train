"""
統一 trainer：以 isinstance(model, nn.Module) 判斷 DL / sklearn 路徑。
export_onnx 已移至各 model 檔案，此處只負責 fit / evaluate。
"""
import numpy as np
from sklearn.metrics import classification_report, f1_score

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