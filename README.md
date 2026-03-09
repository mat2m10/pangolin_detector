# Pangolin Detector

An end-to-end computer vision project for detecting pangolins in images using deep learning object detection.

This repository is designed as a practical wildlife AI tool

## Why this project matters

Pangolins are among the most threatened mammals in the world, and automated detection tools can help support wildlife monitoring, conservation research, and camera-trap image analysis.

## What this project does

The goal of this project is to detect pangolins in still images and return:
- bounding boxes around detected pangolins
- confidence scores for each detection
- visualized prediction outputs for easy inspection

## Project highlights

- End-to-end object detection pipeline
- Pangolin image annotation and preprocessing workflow
- Model training and evaluation scripts
- Inference pipeline for new images
- Simple web app for interactive demos
- GitHub-ready structure for reproducibility and showcase

## Tech stack

- Python
- YOLO-based object detection
- PyTorch
- Streamlit
- Jupyter notebooks

## Repository structure

```text
pangolin_detector/
├── app/                 # Streamlit demo app
├── configs/             # Dataset and training configuration
├── data/                # Raw and processed data
├── examples/            # Example inputs and outputs
├── models/              # Saved weights or exported models
├── notebooks/           # Exploration and experiments
├── scripts/             # Training, prediction, evaluation, preprocessing
├── src/                 # Reusable source code
├── README.md
├── requirements.txt
└── .gitignore