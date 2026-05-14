"""
run_megadetector.py
-------------------
Runs MegaDetector V5 over all frames in data/Frames and saves results.

Usage:
    pip install megadetector
    python run_megadetector.py [--frames-dir PATH] [--output PATH] [--conf FLOAT]

Defaults:
    --frames-dir   ./data/Frames
    --output       ./data/detections.json
    --conf         0.2

Also writes:
    ./data/detections_summary.csv   (one row per frame, joinable with reviews.csv)
"""

import argparse
import csv
import json
from pathlib import Path


def run_detection(frames_dir: Path, output_path: Path, conf_threshold: float):
    try:
        from megadetector.detection.run_detector_batch import (
            load_and_run_detector_batch, write_results_to_file
        )
    except ImportError:
        print("❌ megadetector not installed. Run: pip install megadetector")
        raise

    print("  Loading MegaDetector V5 (weights auto-downloaded on first run)...")
    results = load_and_run_detector_batch(
        model_file="MDV5A",
        image_file_names=str(frames_dir),
        quiet=False,
    )

    # Save full JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_results_to_file(
        results,
        str(output_path),
        relative_path_base=str(frames_dir),
        detector_file="MDV5A",
    )
    print(f"\n  Full results → {output_path}")

    # Write summary CSV
    csv_path = output_path.parent / "detections_summary.csv"
    rows = []

    for img_result in results["images"]:
        rel_path = Path(img_result["file"])
        parts    = rel_path.parts
        sighting = parts[-2] if len(parts) >= 2 else "unknown"
        filename = parts[-1]
        frame_id = f"{sighting}/{filename}"

        detections = img_result.get("detections") or []
        animal_dets = [
            d for d in detections
            if d.get("category") == "1"          # 1 = animal
            and d.get("conf", 0) >= conf_threshold
        ]

        max_conf = max((d.get("conf", 0) for d in animal_dets), default=0.0)
        boxes    = [d.get("bbox", []) for d in animal_dets]

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
        writer = csv.DictWriter(f, fieldnames=[
            "sighting", "filename", "frame_id",
            "animal_detected", "max_confidence", "n_detections", "boxes"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Summary  → {csv_path}")

    detected   = sum(1 for r in rows if r["animal_detected"])
    undetected = len(rows) - detected
    avg_conf   = (
        sum(r["max_confidence"] for r in rows if r["animal_detected"]) / max(detected, 1)
    )

    print(f"\n{'='*50}")
    print(f"  Done!")
    print(f"  Frames processed : {len(rows)}")
    print(f"  Animal detected  : {detected}")
    print(f"  Not detected     : {undetected}")
    print(f"  Avg confidence   : {avg_conf:.3f}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="./data/Frames")
    parser.add_argument("--output",     default="./data/detections.json")
    parser.add_argument("--conf",       default=0.2, type=float)
    args = parser.parse_args()

    frames_dir  = Path(args.frames_dir)
    output_path = Path(args.output)

    print(f"\n{'='*50}")
    print(f"  MegaDetector V5 — Pangolin Detector")
    print(f"  Frames dir : {frames_dir}")
    print(f"  Output     : {output_path}")
    print(f"  Conf       : {args.conf}")
    print(f"{'='*50}\n")

    if not frames_dir.exists():
        print(f"❌ Frames directory not found: {frames_dir}")
        return

    run_detection(frames_dir, output_path, args.conf)


if __name__ == "__main__":
    main()