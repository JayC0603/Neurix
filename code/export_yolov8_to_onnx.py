import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Export a YOLOv8 .pt model to ONNX.")
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "model" / "durain.pt"),
        help="Path to .pt weights.",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Export image size.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "model"),
        help="Directory for copied ONNX model.",
    )
    parser.add_argument("--output-name", default="durain.onnx", help="Output ONNX filename.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    output_dir = Path(args.output_dir)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=False))

    target = output_dir / args.output_name
    target.write_bytes(exported.read_bytes())
    print(f"ONNX model saved: {target}")


if __name__ == "__main__":
    main()
