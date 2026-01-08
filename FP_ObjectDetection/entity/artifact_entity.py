from dataclasses import dataclass
from typing import Any, Dict, Optional

from torch.utils.data.dataloader import DataLoader


@dataclass
class DataIngestionArtifact:
    train_file_path: str

    test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_object: DataLoader

    transformed_test_object: DataLoader

    train_transform_file_path: str

    test_transform_file_path: str



@dataclass
class ModelTrainerArtifact:
    trained_model_path: str


@dataclass
class ODModelTrainerArtifact:
    """Artifacts produced by YOLO object-detection training."""

    best_weights_path: str
    last_weights_path: Optional[str] = None
    training_artifact_dir: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    


@dataclass
class ModelEvaluationArtifact:
    model_accuracy: float


@dataclass
class ODModelEvaluationArtifact:
    """Evaluation artifacts for YOLO object detection."""

    metrics: Dict[str, Any]
    mlflow_run_id: Optional[str] = None


@dataclass
class ModelPusherArtifact:
    # Legacy Bento fields (kept for backward compatibility)
    bentoml_model_name: str = ""
    bentoml_service_name: str = ""

    # New local deployment fields
    pushed_model_path: str = ""
    deployed_models_dir: str = ""
