"""
review_app.py
-------------
Local web app for reviewing pangolin camera trap frames.

Usage:
    pip install flask
    python review_app.py [--frames-dir PATH] [--port PORT]

Defaults:
    --frames-dir  ./data/Frames
    --port        5000

Then open http://localhost:5000 in your browser.

Tags per image:
    Pangolin presence : present / absent / unsure
    Quality           : good / blurry / partial / dark

Exports:
    data/reviews.csv  — one row per reviewed image
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

app = Flask(__name__)

# --- Config (set by main()) ---
FRAMES_DIR = Path("./data/Frames")
REVIEWS_FILE = Path("./data/reviews.csv")

PRESENCE_OPTIONS = ["present", "absent", "unsure"]
QUALITY_OPTIONS  = ["good", "blurry", "partial", "dark"]

# In-memory index of all frames and reviews
frame_index: list[dict] = []   # [{id, sighting, filename, path_rel}, ...]
reviews: dict[str, dict] = {}  # {frame_id: {presence, quality, notes, ts}}


def build_index():
    global frame_index
    frame_index = []
    for sighting_dir in sorted(FRAMES_DIR.iterdir()):
        if not sighting_dir.is_dir():
            continue
        for img in sorted(sighting_dir.glob("*.jpg")) + sorted(sighting_dir.glob("*.JPG")) + sorted(sighting_dir.glob("*.jpeg")):
            frame_index.append({
                "id":       f"{sighting_dir.name}/{img.name}",
                "sighting": sighting_dir.name,
                "filename": img.name,
                "path":     str(img),
            })


def load_reviews():
    global reviews
    reviews = {}
    if REVIEWS_FILE.exists():
        with open(REVIEWS_FILE, newline="") as f:
            for row in csv.DictReader(f):
                reviews[row["id"]] = {
                    "presence": row["presence"],
                    "quality":  row["quality"],
                    "notes":    row.get("notes", ""),
                    "ts":       row.get("ts", ""),
                }


def save_reviews():
    REVIEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEWS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sighting", "filename", "presence", "quality", "notes", "ts"])
        writer.writeheader()
        for frame in frame_index:
            fid = frame["id"]
            if fid in reviews:
                r = reviews[fid]
                writer.writerow({
                    "id":       fid,
                    "sighting": frame["sighting"],
                    "filename": frame["filename"],
                    "presence": r["presence"],
                    "quality":  r["quality"],
                    "notes":    r.get("notes", ""),
                    "ts":       r.get("ts", ""),
                })


# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("review_ui.html")


@app.route("/api/frames")
def api_frames():
    return jsonify({
        "frames":  frame_index,
        "reviews": reviews,
        "total":   len(frame_index),
        "done":    len(reviews),
    })


@app.route("/api/review", methods=["POST"])
def api_review():
    data = request.json
    fid  = data.get("id")
    if not fid:
        return jsonify({"error": "missing id"}), 400

    reviews[fid] = {
        "presence": data.get("presence", ""),
        "quality":  data.get("quality", ""),
        "notes":    data.get("notes", ""),
        "ts":       datetime.utcnow().isoformat(),
    }
    save_reviews()
    return jsonify({"ok": True, "done": len(reviews), "total": len(frame_index)})


@app.route("/api/review/<path:fid>", methods=["DELETE"])
def api_delete_review(fid):
    if fid in reviews:
        del reviews[fid]
        save_reviews()
    return jsonify({"ok": True})


@app.route("/api/export")
def api_export():
    save_reviews()
    return send_file(str(REVIEWS_FILE.resolve()), as_attachment=True, download_name="reviews.csv")


@app.route("/img/<path:rel_path>")
def serve_image(rel_path):
    full = FRAMES_DIR / rel_path
    return send_from_directory(str(full.parent), full.name)


# ── Main ──────────────────────────────────────────────────

def main():
    global FRAMES_DIR, REVIEWS_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="./data/Frames")
    parser.add_argument("--port",       default=5000, type=int)
    args = parser.parse_args()

    FRAMES_DIR   = Path(args.frames_dir)
    REVIEWS_FILE = FRAMES_DIR.parent / "reviews.csv"

    if not FRAMES_DIR.exists():
        print(f"❌ Frames directory not found: {FRAMES_DIR}")
        return

    build_index()
    load_reviews()

    print(f"\n{'='*50}")
    print(f"  Pangolin Review App")
    print(f"  Frames  : {len(frame_index)}")
    print(f"  Reviewed: {len(reviews)}")
    print(f"  Open    : http://localhost:{args.port}")
    print(f"{'='*50}\n")

    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
