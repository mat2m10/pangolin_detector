"""
evaluate.py
-----------
Evaluate trained pangolin weights on a dataset split.

Usage:
    python scripts/evaluate.py [--weights runs/detect/pangolin/weights/best.pt]
                               [--data configs/data.yaml] [--split val]
                               [--imgsz 640] [--conf 0.001] [--dry-run]

Prints mAP50, mAP50-95, precision and recall. Plots and per-class results are
written to runs/val/<name>/.
"""

import argparse
import sys
from pathlib import Path

from common import DEFAULT_DATA, DEFAULT_WEIGHTS, REPO_ROOT, resolve_data_yaml


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a YOLO pangolin detector.")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (low for mAP)")
    p.add_argument("--device", default=None)
    p.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "val")
    p.add_argument("--name", default="pangolin")
    p.add_argument("--dry-run", action="store_true", help="Check inputs, then exit without evaluating")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_yaml = resolve_data_yaml(args.data, require_splits=(args.split,))

    print(f"Weights : {args.weights}")
    print(f"Data    : {args.data} (split: {args.split})")
    if not args.dry_run and not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights} (train first with scripts/train.py)")
    if args.dry_run:
        print("Dry run: not evaluating.")
        return 0

    from ultralytics import YOLO

    val_kwargs = dict(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        project=str(args.project),
        name=args.name,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device

    metrics = YOLO(str(args.weights)).val(**val_kwargs)
    print(f"\nmAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print(f"precision : {metrics.box.mp:.4f}")
    print(f"recall    : {metrics.box.mr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
