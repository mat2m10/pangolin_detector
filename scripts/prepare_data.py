"""
prepare_data.py
---------------
Label cleaning and YOLO train/val split for the pangolin dataset.

This is the logic that used to live in notebooks/data_exploration.ipynb.
All steps work on a dataset folder (default: data/dataset, relative to the
repository root) laid out as:

    data/dataset/
    ├── images/                    # source images (.jpg/.jpeg/.png)
    ├── labels/                    # source YOLO labels (one .txt per image)
    ├── labels_colored/            # written by `colorize` (JSON, one colour per box)
    ├── images_annotated/          # written by `colorize --annotate`
    ├── labels_colored_approved/   # written by the `review` tool
    ├── labels_yolo/               # written by `to-yolo`
    ├── images/{train,val}/        # written by `split`
    ├── labels/{train,val}/        # written by `split`
    └── pangolin.yaml              # written by `split`

Usage (from anywhere):
    python scripts/prepare_data.py colorize [--annotate]
    python scripts/prepare_data.py review          # interactive matplotlib tool
    python scripts/prepare_data.py to-yolo
    python scripts/prepare_data.py stats
    python scripts/prepare_data.py split [--val-fraction 0.2] [--seed 42]

Every command accepts --dataset-dir PATH.
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "dataset"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
CLASS_NAMES = {0: "pangolin"}

COLORS = [
    ("crimson", (220, 20, 60)),
    ("royal_blue", (65, 105, 225)),
    ("forest_green", (34, 139, 34)),
    ("gold", (255, 215, 0)),
    ("dark_orange", (255, 140, 0)),
    ("medium_purple", (147, 112, 219)),
    ("deep_pink", (255, 20, 147)),
    ("turquoise", (64, 224, 208)),
    ("tomato", (255, 99, 71)),
    ("slate_blue", (106, 90, 205)),
    ("lime_green", (50, 205, 50)),
    ("orchid", (218, 112, 214)),
    ("steel_blue", (70, 130, 180)),
    ("coral", (255, 127, 80)),
    ("dark_cyan", (0, 139, 139)),
    ("sandy_brown", (244, 164, 96)),
    ("dodger_blue", (30, 144, 255)),
    ("medium_sea_green", (60, 179, 113)),
    ("hot_pink", (255, 105, 180)),
    ("goldenrod", (218, 165, 32)),
]


# ── Pure helpers (no I/O, easy to test) ───────────────────

def list_images(folder: Path) -> list[Path]:
    """Image files directly inside `folder` (not recursive), sorted."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def yolo_lines_to_entries(lines: list[str]) -> list[dict]:
    """Parse YOLO label lines into coloured annotation dicts (one colour per box)."""
    entries = []
    for idx, line in enumerate(l.strip() for l in lines if l.strip()):
        parts = line.split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
        color_name, color = COLORS[idx % len(COLORS)]
        entries.append({
            "instance_id": idx,
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "color_name": color_name,
            "color_rgb": list(color),
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        })
    return entries


def entries_to_yolo_lines(entries: list[dict]) -> list[str]:
    """Convert annotation dicts back to YOLO label lines."""
    return [
        f'{int(a["class_id"])} {float(a["x_center"]):.6f} {float(a["y_center"]):.6f} '
        f'{float(a["width"]):.6f} {float(a["height"]):.6f}'
        for a in entries
    ]


def split_pairs(pairs: list, val_fraction: float, seed: int) -> tuple[list, list]:
    """Deterministically shuffle and split into (train, val)."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    split_idx = int((1 - val_fraction) * len(pairs))  # same rule as the original notebook
    return pairs[:split_idx], pairs[split_idx:]


# ── Steps ─────────────────────────────────────────────────

def colorize(dataset_dir: Path, annotate: bool = False) -> None:
    labels_dir = dataset_dir / "labels"
    out_dir = dataset_dir / "labels_colored"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    for label_file in label_files:
        entries = yolo_lines_to_entries(label_file.read_text(encoding="utf-8").splitlines())
        (out_dir / label_file.name).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Saved {len(label_files)} colored label files to {out_dir}")

    if annotate:
        annotate_images(dataset_dir)


def annotate_images(dataset_dir: Path) -> None:
    from PIL import Image, ImageDraw

    colored_dir = dataset_dir / "labels_colored"
    out_dir = dataset_dir / "images_annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for image_path in list_images(dataset_dir / "images"):
        label_path = colored_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Skipping {image_path.name}: no label file found.")
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        img_w, img_h = image.size
        for ann in json.loads(label_path.read_text(encoding="utf-8")):
            xc, yc = ann["x_center"] * img_w, ann["y_center"] * img_h
            bw, bh = ann["width"] * img_w, ann["height"] * img_h
            draw.rectangle(
                [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2],
                outline=tuple(ann["color_rgb"]),
                width=3,
            )
        image.save(out_dir / image_path.name)
        n += 1
    print(f"Saved {n} annotated images to {out_dir}")


def review(dataset_dir: Path) -> None:
    """Launch the interactive box-review tool (writes labels_colored_approved/)."""
    tool = REPO_ROOT / "notebooks" / "review_labels_step2.py"
    if dataset_dir.resolve() != DEFAULT_DATASET_DIR.resolve():
        print(f"Note: {tool.name} always works on {DEFAULT_DATASET_DIR}")
    # The tool uses ../data/dataset, so run it from the notebooks folder.
    subprocess.run([sys.executable, str(tool)], cwd=tool.parent, check=True)


def to_yolo(dataset_dir: Path) -> None:
    approved_dir = dataset_dir / "labels_colored_approved"
    out_dir = dataset_dir / "labels_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(approved_dir.glob("*.txt"))  # JSON content, .txt extension
    for json_file in json_files:
        lines = entries_to_yolo_lines(json.loads(json_file.read_text(encoding="utf-8")))
        (out_dir / json_file.name).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    print(f"Converted {len(json_files)} files into YOLO labels at: {out_dir}")


def stats(dataset_dir: Path) -> None:
    images = list_images(dataset_dir / "images")
    yolo_dir = dataset_dir / "labels_yolo"
    non_empty = empty = missing = 0
    for image_path in images:
        label_path = yolo_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing += 1
        elif label_path.read_text(encoding="utf-8").strip():
            non_empty += 1
        else:
            empty += 1
    print(f"Total images: {len(images)}")
    print(f"Non-empty labels: {non_empty}")
    print(f"Empty labels: {empty}")
    print(f"Missing label files: {missing}")

    for split in ("train", "val"):
        labels_dir = dataset_dir / "labels" / split
        if not labels_dir.is_dir():
            continue
        files = sorted(labels_dir.glob("*.txt"))
        ne = sum(1 for f in files if f.read_text(encoding="utf-8").strip())
        print(f"{split} total: {len(files)} non-empty: {ne} empty: {len(files) - ne}")


def split(dataset_dir: Path, val_fraction: float = 0.2, seed: int = 42) -> None:
    import yaml

    images_dir = dataset_dir / "images"
    yolo_dir = dataset_dir / "labels_yolo"

    out_dirs = {
        (kind, part): dataset_dir / kind / part
        for kind in ("images", "labels")
        for part in ("train", "val")
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():  # start from a clean split on every run
            if p.is_file():
                p.unlink()

    pairs = [
        (img, yolo_dir / f"{img.stem}.txt")
        for img in list_images(images_dir)
        if (yolo_dir / f"{img.stem}.txt").exists()
    ]
    print(f"Found {len(pairs)} image-label pairs.")

    train_pairs, val_pairs = split_pairs(pairs, val_fraction, seed)
    for part, part_pairs in (("train", train_pairs), ("val", val_pairs)):
        for image_path, label_path in part_pairs:
            shutil.copy2(image_path, out_dirs[("images", part)] / image_path.name)
            shutil.copy2(label_path, out_dirs[("labels", part)] / label_path.name)

    yaml_path = dataset_dir / "pangolin.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": CLASS_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Saved dataset YAML to: {yaml_path}")


# ── CLI ───────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the pangolin YOLO dataset.")
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
        help="Dataset folder (default: <repo>/data/dataset)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("colorize", help="labels/ -> labels_colored/ (JSON, one colour per box)")
    p.add_argument("--annotate", action="store_true", help="Also draw boxes into images_annotated/")
    sub.add_parser("review", help="Click boxes to drop them -> labels_colored_approved/")
    sub.add_parser("to-yolo", help="labels_colored_approved/ -> labels_yolo/")
    sub.add_parser("stats", help="Count empty / non-empty / missing labels")
    p = sub.add_parser("split", help="Train/val split into images|labels/{train,val} + pangolin.yaml")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = args.dataset_dir
    if not dataset_dir.is_dir():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    if args.command == "colorize":
        colorize(dataset_dir, annotate=args.annotate)
    elif args.command == "review":
        review(dataset_dir)
    elif args.command == "to-yolo":
        to_yolo(dataset_dir)
    elif args.command == "stats":
        stats(dataset_dir)
    elif args.command == "split":
        split(dataset_dir, val_fraction=args.val_fraction, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
