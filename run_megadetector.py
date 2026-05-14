"""
run_megadetector.py
-------------------
Runs MegaDetector V6 over all frames in data/Frames and saves results.

Usage:
    pip install PytorchWildlife
    python run_megadetector.py [--frames-dir PATH] [--output PATH] [--conf FLOAT] [--batch-size INT]

Defaults:
    --frames-dir   ./data/Frames
    --output       ./data/detections.json   (full bounding box results)
    --conf         0.2                      (confidence threshold — low to catch everything)
    --batch-size   16

Also writes:
    ./data/detections_summary.csv           (one row per frame, easy to join with reviews.csv)

Output CSV columns:
    sighting, filename, frame_id, animal_detected, max_confidence, n_detections, boxes
"""

import argparse
import csv
import json
from pathlib import Path


def collect_images(frames_dir: Path) -> list[Path]:
    images = []
    for sighting in sorted(frames_dir.iterdir()):
        if not sighting.is_dir():
            continue
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
            images.extend(sorted(sighting.glob(ext)))
    return images


def run_detection(frames_dir: Path, output_path: Path, conf: float, batch_size: int):
    try:
        from PytorchWildlife.models import detection as pw_detection
    except ImportError:
        print("❌ PytorchWildlife not installed. Run: pip install PytorchWildlife")
        raise

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device  : {device}")

    print("  Loading MegaDetector V6 (weights auto-downloaded on first run)...")
    model = pw_detection.MegaDetectorV6()

    images = collect_images(frames_dir)
    print(f"  Images  : {len(images)}")
    print(f"  Conf    : {conf}")
    print(f"  Batch   : {batch_size}\n")

    # Run batch detection — PytorchWildlife accepts a folder path directly
    print("  Running detection...")
    results = model.batch_image_detection(str(frames_dir), batch_size=batch_size)

    # Save full JSON results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results → {output_path}")

    # Write summary CSV
    csv_path = output_path.parent / "detections_summary.csv"
    rows = []

    for img_path, result in zip(images, results):
        sighting = img_path.parent.name
        filename = img_path.name
        frame_id = f"{sighting}/{filename}"

        # Extract detections above confidence threshold
        detections = result.get("detections", []) if isinstance(result, dict) else []
        animal_dets = [
            d for d in detections
            if d.get("category", "") == "1"   # category 1 = animal in MegaDetector
            and d.get("conf", 0) >= conf
        ]

        max_conf = max((d.get("conf", 0) for d in animal_dets), default=0.0)
        boxes = [d.get("bbox", []) for d in animal_dets]

        rows.append({
            "sighting":        sighting,
            "filename":        filename,
            "frame_id":        frame_id,
            "animal_detected": len(animal_dets) > 0,
            "max_confidence":  round(max_conf, 4),
            "n_detections":    len(animal_dets),
            "boxes":           json.dumps(boxes),
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sighting", "filename", "frame_id",
                                               "animal_detected", "max_confidence",
                                               "n_detections", "boxes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Summary  → {csv_path}")

    # Quick stats
    detected   = sum(1 for r in rows if r["animal_detected"])
    undetected = len(rows) - detected
    print(f"\n{'='*50}")
    print(f"  Done!")
    print(f"  Animal detected : {detected} / {len(rows)} frames")
    print(f"  Not detected    : {undetected} frames")
    if rows:
        avg_conf = sum(r["max_confidence"] for r in rows if r["animal_detected"]) / max(detected, 1)
        print(f"  Avg confidence  : {avg_conf:.3f}")
    print(f"{'='*50}\n")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run MegaDetector V6 on pangolin frames.")
    parser.add_argument("--frames-dir",  default="./data/Frames")
    parser.add_argument("--output",      default="./data/detections.json")
    parser.add_argument("--conf",        default=0.2,  type=float,
                        help="Confidence threshold (default 0.2 — keep low to catch all animals)")
    parser.add_argument("--batch-size",  default=16,   type=int)
    args = parser.parse_args()

    frames_dir  = Path(args.frames_dir)
    output_path = Path(args.output)

    print(f"\n{'='*50}")
    print(f"  MegaDetector V6 — Pangolin Detector")
    print(f"  Frames dir : {frames_dir}")
    print(f"  Output     : {output_path}")
    print(f"{'='*50}")

    if not frames_dir.exists():
        print(f"❌ Frames directory not found: {frames_dir}")
        return

    run_detection(frames_dir, output_path, args.conf, args.batch_size)


if __name__ == "__main__":
    main()
