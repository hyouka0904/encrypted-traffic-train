"""
共用的 ONNX export 工具，供各 model 檔案 import 使用。

  export_sklearn(model, feature_cols, output_path, options=None)
      → sklearn / Pipeline 模型，輸出 label: 字串類別

  export_dl(model, feature_cols, output_path, label_encoder)
      → DL (nn.Module) 模型，輸出 label: int64 index，並寫 label_classes.txt
"""
import os
import sys
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_sklearn(model, feature_cols, output_path, options=None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]

    # skl2onnx 會把 warning 印到 stdout/stderr，全部抑制
    sys.stdout.flush()
    sys.stderr.flush()
    old_out, old_err = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=17,
            options=options or {},
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


def export_dl(model, feature_cols, output_path, label_encoder) -> Path:
    import torch
    import torch.nn as nn

    class _ArgmaxWrapper(nn.Module):
        """logits → argmax，讓 ONNX 輸出 int64 class index（命名為 label）。"""
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            return self.backbone(x).argmax(dim=1)

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
        dynamo=False,
    )

    classes_path = output_path.parent / "label_classes.txt"
    classes_path.write_text("\n".join(label_encoder.classes_))
    print(f"[export] label_classes.txt → {classes_path}")
    print(f"[export] {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path