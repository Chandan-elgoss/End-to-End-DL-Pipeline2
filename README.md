# End-to-End Floor Plan Object Detection (YOLO + MLflow)

This project has been refactored from a Lung X-ray classifier into a production-style pipeline for detecting floorplan objects (e.g., bed, sofa, etc.) using Ultralytics YOLO, with MLflow tracking and a simple FastAPI UI.

## Structure

FP_ObjectDetection/components: training, evaluation (logs to MLflow), and local deployment (copies best.pt to deployed_models/)
FP_ObjectDetection/entity: config and artifact dataclasses
FP_ObjectDetection/pipeline: `TrainPipeline` orchestrating train → evaluate → push

- datasets/: YOLO-format dataset (see datasets/FURNITURE_DETECTION.v1/data.yaml)
- app.py: FastAPI app for local inference UI
  Edit constants in [FP_ObjectDetection/constant/training_pipeline/**init**.py](FP_ObjectDetection/constant/training_pipeline/__init__.py):

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Configure

Edit constants in [Xray/constant/training_pipeline/**init**.py](Xray/constant/training_pipeline/__init__.py):

- `DATASET_YAML_PATH`: path to your YOLO data.yaml
- `YOLO_BASE_MODEL`: e.g., `yolov8n.pt`
- `EPOCHS`, `IMAGE_SIZE`, `BATCH`: set your training hyperparameters
- MLflow server: defaults to `http://127.0.0.1:5000`

Start MLflow UI (optional):

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

## Train, Evaluate, Push

```bash
python main.py
```

Artifacts will be written under `artifacts/<timestamp>/...`, best weights copied to `deployed_models/best.pt`.

## Run UI

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and upload an image.

## Notes

- Legacy data ingestion/transformation and BentoML classifier files are kept but marked deprecated; they are not used in this pipeline.
