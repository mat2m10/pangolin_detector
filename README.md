# Pangolin Detector

Tooling for building a pangolin detector for camera-trap footage: frame
extraction, MegaDetector pre-filtering, a local review UI and label-cleaning
steps that produce a YOLO-format train/val dataset.

Pangolins are among the most threatened mammals in the world, and automated
detection helps wildlife monitoring and camera-trap analysis.

## Status

The data-preparation pipeline works. **No detector has been trained yet**:
`scripts/train.py`, `scripts/predict.py`, `scripts/evaluate.py`,
`scripts/prepare_data.py`, `configs/data.yaml` and `app/app.py` are empty
placeholders.

## Pipeline

All commands are run from the repository root. Raw data is expected under
`data/` (gitignored).

1. **Extract frames** from the camera-trap clips (needs `ffmpeg` on PATH):

   ```bash
   python extract_frames.py --sightings-dir data/Sightings --output-dir data/Frames --fps 1
   ```

   One folder per sighting is written to `data/Frames/`. Compiled
   `SIGHTING_*.mp4` files are skipped, JPGs are copied as-is, and sightings
   that were already extracted are skipped. Add `--dry-run` to preview.

2. **Pre-filter with MegaDetector V5** (weights download on first run):

   ```bash
   python run_megadetector.py --frames-dir data/Frames --output data/detections.json --conf 0.2
   ```

   This writes `data/detections.json` and `data/detections_summary.csv`
   (one row per frame, joinable with the review CSV on `frame_id`).

3. **Review frames by hand** in a local web app:

   ```bash
   python review_app.py --frames-dir data/Frames --port 5000
   # open http://localhost:5000
   ```

   Tag each frame for pangolin presence (present / absent / unsure) and
   quality (good / blurry / partial / dark). Results are saved to
   `data/reviews.csv`.

4. **Clean the bounding-box labels and build the YOLO split** in
   `notebooks/data_exploration.ipynb`. It works on `data/dataset/`
   (`images/` plus YOLO `labels/`), and:
   - colours each box and writes `labels_colored/` and `images_annotated/`;
   - launches `notebooks/review_labels_step2.py`, a matplotlib tool where
     you click boxes to drop them, which writes `labels_colored_approved/`;
   - converts the approved labels back to YOLO format in `labels_yolo/`;
   - makes an 80/20 split into `images/{train,val}` and
     `labels/{train,val}`, and writes `data/dataset/pangolin.yaml`.

   Run the notebook from the `notebooks/` folder, because it uses
   `../data/dataset` paths.

5. **Train**: not implemented yet. The dataset from step 4 is in standard
   Ultralytics format, so `yolo detect train data=data/dataset/pangolin.yaml model=yolo11n.pt`
   is the intended starting point.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`review_app.py` needs `flask` and `run_megadetector.py` needs `megadetector`.
Install them separately if they are not in `requirements.txt` yet.

## Repository structure

```text
pangolin_detector/
├── extract_frames.py      # step 1: video -> frames (ffmpeg)
├── run_megadetector.py    # step 2: MegaDetector V5 pre-filter
├── review_app.py          # step 3: Flask review app (serves review_ui.html)
├── review_ui.html
├── notebooks/
│   ├── data_exploration.ipynb   # step 4: label cleaning + YOLO split
│   ├── review_labels_step1.py   # single-image box review prototype
│   ├── review_labels_step2.py   # box review tool used by the notebook
│   └── sighting_labels.ipynb    # scratch
├── scripts/               # placeholders (empty): train / predict / evaluate / prepare_data
├── configs/data.yaml      # placeholder (empty)
├── app/app.py             # placeholder (empty)
├── requirements.txt
└── Makefile               # `make clean` removes Windows *Zone.Identifier files
```

`data/` is gitignored and is not part of the repository.

## Tech stack

Python, ffmpeg, MegaDetector V5, Flask, Jupyter, and Ultralytics YOLO with
PyTorch for the training step.
