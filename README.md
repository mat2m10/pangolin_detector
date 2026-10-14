# Pangolin Detector

Tooling for building a pangolin detector for camera-trap footage: frame
extraction, MegaDetector pre-filtering, a local review UI and label-cleaning
steps that produce a YOLO-format train/val dataset.

Pangolins are among the most threatened mammals in the world, and automated
detection helps wildlife monitoring and camera-trap analysis.

## Status

The data-preparation pipeline (steps 1-4) and the training, evaluation and
prediction scripts (step 5) are in place. No trained weights are committed:
train your own with `scripts/train.py` (results go to the git-ignored
`runs/` folder).

## Pipeline

Run the commands from the repository root. Raw data is expected under
`data/` (git-ignored). The tools in steps 1-3 resolve their default `./data/...`
paths against the current directory; `scripts/prepare_data.py`,
`train.py`, `evaluate.py` and `predict.py` resolve theirs against the
repository root and work from anywhere.

1. **Extract frames** from the camera-trap clips (needs `ffmpeg` on PATH):

   ```bash
   python scripts/extract_frames.py --sightings-dir data/Sightings --output-dir data/Frames --fps 1
   ```

   One folder per sighting is written to `data/Frames/`. Compiled
   `SIGHTING_*.mp4` files are skipped, JPGs are copied as-is, and sightings
   that were already extracted are skipped. Add `--dry-run` to preview.

2. **Pre-filter with MegaDetector V5** (weights download on first run):

   ```bash
   python scripts/run_megadetector.py --frames-dir data/Frames --output data/detections.json --conf 0.2
   ```

   This writes `data/detections.json` and `data/detections_summary.csv`
   (one row per frame, joinable with the review CSV on `frame_id`).

3. **Review frames by hand** in a local web app:

   ```bash
   python tools/review/review_app.py --frames-dir data/Frames --port 5000
   # open http://localhost:5000
   ```

   Tag each frame for pangolin presence (present / absent / unsure) and
   quality (good / blurry / partial / dark). Results are saved to
   `data/reviews.csv`.

4. **Clean the bounding-box labels and build the YOLO split** with
   `scripts/prepare_data.py`. It works on `data/dataset/` (`images/` plus
   YOLO `labels/`; change it with `--dataset-dir PATH` before the command):

   ```bash
   python scripts/prepare_data.py colorize --annotate   # labels/ -> labels_colored/ + images_annotated/
   python scripts/prepare_data.py review                # click boxes to drop them -> labels_colored_approved/
   python scripts/prepare_data.py to-yolo               # -> labels_yolo/
   python scripts/prepare_data.py split --val-fraction 0.2 --seed 42
   python scripts/prepare_data.py stats                 # empty / non-empty / missing labels
   ```

   `review` opens `notebooks/review_labels_step2.py`, a matplotlib tool.
   `split` copies the image/label pairs into `images/{train,val}` and
   `labels/{train,val}` and writes `data/dataset/pangolin.yaml`.
   `notebooks/data_exploration.ipynb` runs the same steps in order.

5. **Train, evaluate and predict** with Ultralytics YOLO, using
   `configs/data.yaml` (points at the split from step 4):

   ```bash
   python scripts/train.py --model yolo11n.pt --epochs 100   # -> runs/detect/pangolin/weights/best.pt
   python scripts/evaluate.py --split val                    # mAP50, mAP50-95, precision, recall
   python scripts/predict.py path/to/images_or_video         # -> runs/predict/pangolin/
   ```

   Base weights such as `yolo11n.pt` are downloaded by Ultralytics on first
   use. Each script has `--help` and a `--dry-run` that checks the config
   and dataset without training or predicting.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins everything, including `flask` (review app),
`megadetector` (step 2) and `ultralytics`/`torch` (step 5). `ffmpeg` must be
installed separately.

## Development

```bash
pip install ruff pytest
ruff check .
pytest -q
```

CI (`.github/workflows/ci.yml`) runs the same on pushes to master and on pull requests.

## Repository structure

```text
pangolin_detector/
├── scripts/
│   ├── extract_frames.py     # step 1: video -> frames (ffmpeg)
│   ├── run_megadetector.py   # step 2: MegaDetector V5 pre-filter
│   ├── prepare_data.py       # step 4: label cleaning + YOLO split
│   ├── train.py              # step 5: train
│   ├── evaluate.py           # step 5: evaluate
│   ├── predict.py            # step 5: predict
│   └── common.py             # shared helpers for train / evaluate / predict
├── tools/review/
│   ├── review_app.py         # step 3: Flask review app
│   └── review_ui.html
├── configs/data.yaml         # Ultralytics dataset config
├── notebooks/
│   ├── data_exploration.ipynb    # runs the step 4 commands in order
│   ├── review_labels_step1.py    # single-image box review prototype
│   ├── review_labels_step2.py    # box review tool used by `prepare_data.py review`
│   └── sighting_labels.ipynb     # scratch
├── tests/                    # pytest smoke tests
├── requirements.txt
└── Makefile                  # `make clean` removes Windows *Zone.Identifier files
```

`data/` and `runs/` are git-ignored and are not part of the repository.

## Tech stack

Python, ffmpeg, MegaDetector V5, Flask, Jupyter, and Ultralytics YOLO with
PyTorch for the training step.
