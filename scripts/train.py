"""
train.py
--------
Fine-tune an Ultralytics YOLO detector on the pangolin dataset.

Usage:
    python scripts/train.py [--data configs/data.yaml] [--model yolo11n.pt]
                            [--epochs 100] [--imgsz 640] [--batch 16]
                            [--device 0] [--name pangolin] [--dry-run]

The base weights (e.g. yolo11n.pt) are downloaded by Ultralytics on first use.
Results go to runs/detect/<name>/ (best weights: weights/best.pt). The final
validation metrics are printed at the end.
"""

import argparse
import sys
from pathlib import Path

from common import DEFAULT_DATA, REPO_ROOT, resolve_data_yaml


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a YOLO pangolin detector.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML (default: configs/data.yaml)")
    p.add_argument("--model", default="yolo11n.pt", help="Base weights or model YAML (default: yolo11n.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="Batch size (-1 = auto)")
    p.add_argument("--device", default=None, help="e.g. 0, 0,1 or cpu (default: auto)")
    p.add_argument("--patience", type=int, default=50, help="Early-stopping patience in epochs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "detect")
    p.add_argument("--name", default="pangolin", help="Run name (folder under --project)")
    p.add_argument("--dry-run", action="store_true", help="Check config and dataset, then exit without training")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_yaml = resolve_data_yaml(args.data)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"Model : {args.model}")
    for k, v in train_kwargs.items():
        print(f"{k:<8}: {v}")

    if args.dry_run:
        print("Dry run: not training.")
        return 0

    from ultralytics import YOLO  # imported late so --help / --dry-run stay fast

    model = YOLO(args.model)
    model.train(**train_kwargs)

    # Validate the best checkpoint and report the headline numbers.
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch if args.batch > 0 else 16)
    print(f"\nBest weights : {model.trainer.best}")
    print(f"mAP50        : {metrics.box.map50:.4f}")
    print(f"mAP50-95     : {metrics.box.map:.4f}")
    print(f"precision    : {metrics.box.mp:.4f}")
    print(f"recall       : {metrics.box.mr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
