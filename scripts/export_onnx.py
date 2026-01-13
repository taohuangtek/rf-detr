import argparse
from rfdetr import RFDETRSmall

def main():
    parser = argparse.ArgumentParser(description="RF-DETR Fixed-Size ONNX Export Tool")

    parser.add_argument(
        "--weights",
        type=str,
        default="rf-detr-small.pth",
        help="Path to the pretrain weights"
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=80,
        help="Number of classes"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Fixed input resolution (e.g., 512 or 640)"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Perform ONNX simplification"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX opset version"
    )

    args = parser.parse_args()

    print(f"Initializing model with resolution: {args.img_size}x{args.img_size}")
    model = RFDETRSmall(
        pretrain_weights=args.weights,
        num_classes=args.classes,
        num_windows=1,
        resolution=args.img_size  # This sets the static input shape
    )

    print(f"Exporting model: simplify={args.simplify}, opset={args.opset}")
    model.export(
        simplify=args.simplify,
        opset_version=args.opset
    )

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
