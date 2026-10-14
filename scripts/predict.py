"""
predict.py
----------
Run trained pangolin weights on images, folders or videos.

Usage:
    python scripts/predict.py SOURCE [--weights runs/detect/pangolin/weights/best.pt]
                              [--conf 0.25] [--imgsz 640] [--no-save] [--dry-run]

SOURCE is anything Ultralytics accepts: an image, a folder, a glob or a video.
Annotated outputs and YOLO-format .txt detections are written to
runs/predict/<name>/. A one-line summary is printed per image.
"""

import argparse
import sys
from pathlib import Path

from common import DEFAULT_WEIGHTS, REPO_ROOT


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Detect pangolins with trained YOLO weights.")
    p.add_argument("source", help="Image, folder, glob or video")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "predict")
    p.add_argument("--name", default="pangolin")
    p.add_argument("--no-save", action="store_true", help="Do not write annotated images / label files")
    p.add_argument("--dry-run", action="store_true", help="Check inputs, then exit without predicting")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Weights : {args.weights}")
    print(f"Source  : {args.source}")
    if args.dry_run:
        print("Dry run: not predicting.")
        return 0
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights} (train first with scripts/train.py)")

    from ultralytics import YOLO

    predict_kwargs = dict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=not args.no_save,
        save_txt=not args.no_save,
        save_conf=True,
        project=str(args.project),
        name=args.name,
        stream=True,  # iterate lazily so large folders/videos do not fill memory
    )
    if args.device is not None:
        predict_kwargs["device"] = args.device

    n_images = n_with_pangolin = 0
    for result in YOLO(str(args.weights)).predict(**predict_kwargs):
        n_images += 1
        confs = result.boxes.conf.tolist() if result.boxes is not None else []
        if confs:
            n_with_pangolin += 1
        best = f"{max(confs):.2f}" if confs else "-"
        print(f"{Path(result.path).name}: {len(confs)} pangolin(s), max conf {best}")

    print(f"\n{n_with_pangolin}/{n_images} images with at least one detection")
    if not args.no_save and n_images:
        print(f"Outputs: {result.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
