"""Training pipeline constants.

This repository started as a Lung X-ray *classification* pipeline.
It has been refactored to support *YOLO object detection* for floor-plan
objects (e.g., bed, sofa, etc.) with MLflow tracking.

Notes
-----
- Data ingestion / transformation (S3 + torchvision) are intentionally
	*not used* for the current object-detection pipeline.
- Keep placeholders empty where you want to feed your own requirements.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# Artifacts
ARTIFACT_DIR: str = "artifacts"


# ===== Object Detection (Ultralytics YOLO) =====
# Path to your YOLO dataset YAML (train/val paths + class names).
# Example in this repo: datasets/FURNITURE_DETECTION.v1/data.yaml
DATASET_YAML_PATH: str = ""  # set me

# Base model to fine-tune (e.g., 'yolov8n.pt', 'yolov8s.pt').
YOLO_BASE_MODEL: str = ""  # set me

# Training hyperparameters
EPOCHS: Optional[int] = None  # set me
IMAGE_SIZE: Optional[int] = None  # set me (e.g., 640)
BATCH: Optional[int] = None  # set me
CONF_THRESHOLD: float = 0.25
IOU_THRESHOLD: float = 0.45


# ===== MLflow =====
MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME: str = "floorplan-object-detection"
MLFLOW_RUN_NAME: str = ""  # optional


# ===== Local deployment =====
# A simple, local “promotion” step: copy best weights here.
DEPLOYED_MODELS_DIR: str = "deployed_models"
DEPLOYED_MODEL_FILENAME: str = "best.pt"
