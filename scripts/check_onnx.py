from __future__ import annotations

import sys
import pathlib

# onnx & related imports
import onnx


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_onnx.py <model1.onnx> [model2.onnx ...]")
        return 2

    for model_path in map(pathlib.Path, sys.argv[1:]):
        model = onnx.load(str(model_path))
        opsets = [(opset.domain, opset.version) for opset in model.opset_import]
        print(f"{model_path} opsets: {opsets}")

        onnx.checker.check_model(model)
        print("OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
