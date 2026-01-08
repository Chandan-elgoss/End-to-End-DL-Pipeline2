import os
from dataclasses import dataclass

from Xray.constant.training_pipeline import (
    ARTIFACT_DIR,
    BATCH,
    CONF_THRESHOLD,
    DATASET_YAML_PATH,
    DEPLOYED_MODEL_FILENAME,
    DEPLOYED_MODELS_DIR,
    EPOCHS,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_RUN_NAME,
    MLFLOW_TRACKING_URI,
    TIMESTAMP,
    YOLO_BASE_MODEL,
)


@dataclass
class DataIngestionConfig:
    """DEPRECATED for the object-detection pipeline.

    Kept to avoid breaking older flows, but it is not used when training
    YOLO floor-plan object detection.
    """

    def __init__(self):
        # Intentionally left as-is (legacy). If you still want S3 ingestion,
        # reintroduce the required constants in Xray.constant.training_pipeline.
        self.s3_data_folder: str = ""
        self.bucket_name: str = ""
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
        self.data_path: str = os.path.join(self.artifact_dir, "data_ingestion")
        self.train_data_path: str = os.path.join(self.data_path, "train")
        self.test_data_path: str = os.path.join(self.data_path, "test")



@dataclass
class DataTransformationConfig:
    """DEPRECATED for the object-detection pipeline.

    YOLO uses dataset YAML + its own dataloaders. This configuration is kept
    only for backward compatibility.
    """

    def __init__(self):
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP, "data_transformation")
        self.train_transforms_file: str = os.path.join(self.artifact_dir, "train_transforms.pkl")
        self.test_transforms_file: str = os.path.join(self.artifact_dir, "test_transforms.pkl")




@dataclass
class ModelTrainerConfig:
    """Object detection trainer configuration (Ultralytics YOLO)."""

    def __init__(self):
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_training")
        self.dataset_yaml_path: str = DATASET_YAML_PATH
        self.base_model: str = YOLO_BASE_MODEL

        # Leave these as None in constants; set them before running.
        self.epochs = EPOCHS
        self.image_size = IMAGE_SIZE
        self.batch = BATCH

        self.conf_threshold: float = CONF_THRESHOLD
        self.iou_threshold: float = IOU_THRESHOLD

        # Where we expect YOLO to write weights under artifact_dir.
        self.weights_dir: str = os.path.join(self.artifact_dir, "weights")
        self.best_weights_path: str = os.path.join(self.weights_dir, "best.pt")
        self.last_weights_path: str = os.path.join(self.weights_dir, "last.pt")
        
@dataclass
class ModelEvaluationConfig:
    """Object detection evaluation configuration (Ultralytics YOLO + MLflow)."""

    def __init__(self):
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_evaluation")
        self.dataset_yaml_path: str = DATASET_YAML_PATH
        self.conf_threshold: float = CONF_THRESHOLD
        self.iou_threshold: float = IOU_THRESHOLD

        # MLflow
        self.mlflow_tracking_uri: str = MLFLOW_TRACKING_URI
        self.mlflow_experiment_name: str = MLFLOW_EXPERIMENT_NAME
        self.mlflow_run_name: str = MLFLOW_RUN_NAME

# Model Pusher Configurations
@dataclass
class ModelPusherConfig:
    def __init__(self):
        # Local-only “deployment”: copy weights to a stable location.
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_pusher")
        self.deployed_models_dir: str = DEPLOYED_MODELS_DIR
        self.deployed_model_filename: str = DEPLOYED_MODEL_FILENAME

