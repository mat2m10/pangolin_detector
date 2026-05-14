"""
extract_frames.py
-----------------
Extracts frames from all MP4 videos in the Sightings dataset.

Usage:
    python extract_frames.py [--sightings-dir PATH] [--output-dir PATH] [--fps N] [--dry-run]

Defaults:
    --sightings-dir  ./data/Sightings
    --output-dir     ./data/Frames
    --fps            1          (1 frame per second)
    --dry-run        False

Output structure mirrors the sightings layout:
    data/Frames/
    └── SIGHTING_T19_1/
        ├── DEPLOYMENT_T19_2__2024-06-15__00-24-07_f0001.jpg
        └── ...

Rules:
    - Compiled sightings (e.g. SIGHTING_T8_3.mp4) are SKIPPED to avoid duplicates.
    - Raw MP4 clips (e.g. IMAG0014.MP4) are extracted.
    - Existing JPG/JPEG images in a sighting folder are copied as-is (no re-extraction needed).
    - Already-extracted sightings are skipped on re-run (idempotent).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


COMPILED_SUFFIXES = {".mp4"}  # lowercase compiled file check
SKIP_PATTERN = "SIGHTING_"    # compiled files start with the sighting folder name


def is_compiled_mp4(video_path: Path) -> bool:
    """Return True if this MP4 is a compiled sighting file (skip it)."""
    return (
        video_path.suffix.lower() == ".mp4"
        and video_path.stem.upper().startswith(SKIP_PATTERN)
    )


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_frames(video_path: Path, output_dir: Path, fps: int, dry_run: bool) -> int:
    """Extract frames from a single video. Returns number of frames extracted."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    output_pattern = output_dir / f"{stem}_f%04d.jpg"

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",          # JPEG quality: 2 = high quality, 31 = lowest
        "-loglevel", "error",
        str(output_pattern),
    ]

    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠️  ffmpeg error on {video_path.name}: {result.stderr.strip()}")
        return 0

    extracted = list(output_dir.glob(f"{stem}_f*.jpg"))
    return len(extracted)


def copy_image(img_path: Path, output_dir: Path, dry_run: bool):
    """Copy an existing JPG image into the output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / img_path.name
    if dest.exists():
        return
    if dry_run:
        print(f"  [dry-run] would copy: {img_path.name}")
        return
    shutil.copy2(img_path, dest)


def process_sightings(sightings_dir: Path, output_dir: Path, fps: int, dry_run: bool):
    sighting_folders = sorted([d for d in sightings_dir.iterdir() if d.is_dir()])

    total_sightings = len(sighting_folders)
    total_videos = 0
    total_skipped_compiled = 0
    total_frames = 0
    total_images_copied = 0
    already_done = 0

    print(f"\n{'='*60}")
    print(f"  Pangolin Frame Extractor")
    print(f"  Source : {sightings_dir}")
    print(f"  Output : {output_dir}")
    print(f"  FPS    : {fps}")
    print(f"  Mode   : {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    for i, sighting in enumerate(sighting_folders, 1):
        out_sighting = output_dir / sighting.name

        # Idempotency check: if output folder exists and has files, skip
        if out_sighting.exists() and any(out_sighting.iterdir()):
            print(f"[{i:02}/{total_sightings}] {sighting.name} — already extracted, skipping")
            already_done += 1
            continue

        print(f"[{i:02}/{total_sightings}] {sighting.name}")

        files = list(sighting.iterdir())
        videos = [f for f in files if f.suffix.upper() in {".MP4"}]
        images = [f for f in files if f.suffix.upper() in {".JPG", ".JPEG"}]

        # Copy existing images
        for img in images:
            copy_image(img, out_sighting, dry_run)
            total_images_copied += 1
            print(f"  📷 copied {img.name}")

        # Process videos
        for vid in videos:
            if is_compiled_mp4(vid):
                print(f"  ⏭️  skipped compiled: {vid.name}")
                total_skipped_compiled += 1
                continue

            print(f"  🎬 extracting from {vid.name} at {fps} fps...")
            n = extract_frames(vid, out_sighting, fps, dry_run)
            total_videos += 1
            total_frames += n
            if not dry_run:
                print(f"     → {n} frames saved")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Sightings processed : {total_sightings - already_done} (skipped {already_done} already done)")
    print(f"  Videos extracted    : {total_videos} ({total_skipped_compiled} compiled skipped)")
    print(f"  Frames extracted    : {total_frames}")
    print(f"  Images copied       : {total_images_copied}")
    print(f"  Output              : {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from pangolin camera trap videos.")
    parser.add_argument("--sightings-dir", default="./data/Sightings", help="Path to Sightings folder")
    parser.add_argument("--output-dir",    default="./data/Frames",    help="Where to write extracted frames")
    parser.add_argument("--fps",           default=1, type=int,         help="Frames per second to extract (default: 1)")
    parser.add_argument("--dry-run",       action="store_true",         help="Preview without writing any files")
    args = parser.parse_args()

    if not check_ffmpeg():
        print("❌ ffmpeg not found. Please install it: https://ffmpeg.org/download.html")
        sys.exit(1)

    sightings_dir = Path(args.sightings_dir)
    if not sightings_dir.exists():
        print(f"❌ Sightings directory not found: {sightings_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    process_sightings(sightings_dir, output_dir, args.fps, args.dry_run)


if __name__ == "__main__":
    main()