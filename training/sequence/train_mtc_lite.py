"""Train MTC-lite on packet-sequence NPZ datasets.

Separate entry point from training/main.py (ARFF tabular path).
Does not export ONNX yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

from models.mtc_lite import build  # noqa: E402

XGBOOST_MACRO_F1_BASELINE = 0.8865


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MTC-lite on a packet-sequence NPZ dataset.",
    )
    parser.add_argument(
        "--data",
        default="data/sequence/iscx_vpn_sequence.npz",
        help="Input NPZ path (default: data/sequence/iscx_vpn_sequence.npz)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory for checkpoints and results JSON (default: models)",
    )
    parser.add_argument(
        "--artifact-name",
        default="mtc_lite",
        help="Base name for saved artifacts (default: mtc_lite)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--min-flows-for-real-result",
        type=int,
        default=100,
        help="Below this flow count, results are marked as smoke tests only",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_arg)


def load_sequence_npz(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not data_path.is_file():
        raise SystemExit(f"Dataset not found: {data_path}")

    npz = np.load(data_path, allow_pickle=True)
    if "X" not in npz or "y" not in npz:
        raise SystemExit(f"NPZ must contain 'X' and 'y': {data_path}")

    x = npz["X"].astype("float32")
    y = np.asarray(npz["y"]).astype(str)
    feature_names = np.asarray(npz["feature_names"]) if "feature_names" in npz else np.array([])

    if "classes" in npz:
        classes = [str(label) for label in npz["classes"].tolist()]
    else:
        classes = sorted(np.unique(y).tolist())

    if x.ndim != 3:
        raise SystemExit(
            f"X must be 3D [num_flows, max_packets, num_features], got shape {x.shape}"
        )
    if len(y) != x.shape[0]:
        raise SystemExit(
            f"y length ({len(y)}) must match num_flows ({x.shape[0]})"
        )

    return x, y, feature_names, classes


def can_stratify(y: np.ndarray) -> bool:
    counts = Counter(y.tolist())
    return all(count >= 2 for count in counts.values())


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if can_stratify(y):
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    print(
        "[warn] Not enough samples per class for stratified split; "
        "using random split without stratify."
    )
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    label_encoder: LabelEncoder,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    y_int = label_encoder.transform(y_train).astype(np.int64)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_int, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    log_every = max(1, epochs // 5)  # every 20% of training

    print(f"[train] MTCLite  epochs={epochs}  device={device}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)

        if epoch % log_every == 0 or epoch == epochs:
            avg_loss = total_loss / len(loader.dataset)
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    model = model.cpu()
    print("[train] done")
    return model


def evaluate_model(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> tuple[float, float, str]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
        y_pred = label_encoder.inverse_transform(logits.argmax(dim=1).numpy())

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, macro_f1, report


def build_results_note(smoke_test: bool, macro_f1: float) -> str:
    if smoke_test:
        return (
            "Smoke test only: dataset has too few flows for a meaningful benchmark. "
            f"XGBoost baseline macro F1 remains {XGBOOST_MACRO_F1_BASELINE:.4f}."
        )
    if macro_f1 > XGBOOST_MACRO_F1_BASELINE:
        return (
            f"Sequence model macro F1 ({macro_f1:.4f}) exceeds XGBoost baseline "
            f"({XGBOOST_MACRO_F1_BASELINE:.4f}); further validation required."
        )
    return (
        f"Sequence model macro F1 ({macro_f1:.4f}) did not beat XGBoost baseline "
        f"({XGBOOST_MACRO_F1_BASELINE:.4f})."
    )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, feature_names, classes = load_sequence_npz(data_path)
    num_flows = x.shape[0]
    smoke_test = num_flows < args.min_flows_for_real_result

    print(f"[data] flows={num_flows}  X shape={tuple(x.shape)}")
    if feature_names.size:
        print(f"[data] features={feature_names.tolist()}")
    print(f"[data] classes={classes}")

    if smoke_test:
        print(
            "SMOKE TEST ONLY: dataset has too few flows for a meaningful result."
        )

    x_train, x_test, y_train, y_test = split_data(
        x, y, args.test_size, args.random_state
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    model_params = {
        "max_packets": x.shape[1],
        "num_packet_features": x.shape[2],
        "hidden_dim": 64,
        "cnn_channels": 64,
        "transformer_layers": 2,
        "transformer_heads": 4,
        "dropout": 0.2,
    }
    model = build(model_params, n_features=x.shape[2], n_classes=len(classes))

    device = resolve_device(args.device)
    model = train_model(
        model,
        x_train,
        y_train,
        label_encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    accuracy, macro_f1, report = evaluate_model(model, x_test, y_test, label_encoder)
    print("\n[evaluate] Classification Report:")
    print(report)
    print(f"[evaluate] accuracy={accuracy:.4f}  macro_f1={macro_f1:.4f}")

    checkpoint_path = output_dir / f"{args.artifact_name}_sequence.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[save] {checkpoint_path}")

    results = {
        "model": "mtc_lite",
        "type": "sequence_dl",
        "data": str(data_path),
        "num_flows": int(num_flows),
        "x_shape": [int(dim) for dim in x.shape],
        "classes": [str(label) for label in classes],
        "accuracy": float(round(accuracy, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "smoke_test": bool(smoke_test),
        "note": str(build_results_note(smoke_test, macro_f1)),
    }
    results_path = output_dir / f"{args.artifact_name}_sequence_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {results_path}")


if __name__ == "__main__":
    main()
