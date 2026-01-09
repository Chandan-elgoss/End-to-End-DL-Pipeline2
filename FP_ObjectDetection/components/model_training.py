 
import os
import sys
from pathlib import Path

from ultralytics import YOLO
from typing import Optional

from FP_ObjectDetection.entity.artifact_entity import ODModelTrainerArtifact
from FP_ObjectDetection.entity.config_entity import ModelTrainerConfig
from FP_ObjectDetection.exception import XRayException
from FP_ObjectDetection.logger import logging


class ModelTrainer:
    """YOLO (Ultralytics) trainer for floor-plan object detection.

    Replaces legacy Lung X-ray classifier training.
    - No DataIngestion / DataTransformation dependency.
    - Expects a YOLO dataset YAML (train/val paths + class names).
    """

    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def _validate_config(self) -> None:
        missing = []
        if not self.model_trainer_config.dataset_yaml_path:
            missing.append("DATASET_YAML_PATH")
        if not self.model_trainer_config.base_model:
            missing.append("YOLO_BASE_MODEL")
        if self.model_trainer_config.epochs is None:
            missing.append("EPOCHS")
        if self.model_trainer_config.image_size is None:
            missing.append("IMAGE_SIZE")
        if self.model_trainer_config.batch is None:
            missing.append("BATCH")
        if missing:
            raise ValueError("Missing required training constants: " + ", ".join(missing))

    def initiate_model_trainer(self, mlflow_run_id: Optional[str] = None) -> ODModelTrainerArtifact:
        """Train YOLO and return best/last weights paths."""
        logging.info("Entered initiate_model_trainer (YOLO)")

        try:
            self._validate_config()
            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            model = YOLO(self.model_trainer_config.base_model)

            # Verbose per-epoch logging via Ultralytics callbacks (prints losses/metrics)
            def _on_fit_epoch_end(trainer):
                try:
                    epoch = getattr(trainer, "epoch", None)
                    epochs = getattr(trainer, "epochs", None)
                    loss_items = getattr(trainer, "loss_items", None) or getattr(trainer, "tloss", None)
                    metrics = getattr(trainer, "metrics", None)

                    parts = []
                    if isinstance(loss_items, (list, tuple)):
                        parts.append("loss_items=[" + ", ".join(f"{x:.4f}" for x in loss_items) + "]")
                    elif isinstance(loss_items, (int, float)):
                        parts.append(f"loss={loss_items:.4f}")

                    # Common YOLOv8 keys; log if present
                    if isinstance(metrics, dict) and metrics:
                        for k in ("box_loss", "cls_loss", "dfl_loss", "mAP50", "mAP50-95"):
                            v = metrics.get(k)
                            if isinstance(v, (int, float)):
                                parts.append(f"{k}={v:.4f}")

                    if epoch is not None and epochs is not None:
                        prefix = f"Epoch {epoch + 1}/{epochs}"
                    elif epoch is not None:
                        prefix = f"Epoch {epoch + 1}"
                    else:
                        prefix = "Epoch end"

                    logging.info("%s - %s", prefix, " ".join(parts) if parts else "(no metrics)")
                except Exception:
                    # Never break training due to logging
                    pass

            try:
                model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
            except Exception:
                # Older/newer versions may differ; ignore if unavailable
                pass

            # Ultralytics will create a run folder under `project/name`.
            # We set `project` to our artifact dir and `name` to '.' so weights land in:
            # artifacts/<timestamp>/model_training/weights/best.pt
            dataset_yaml = str(Path(self.model_trainer_config.dataset_yaml_path).resolve())
            _ = model.train(
                data=dataset_yaml,
                epochs=int(self.model_trainer_config.epochs),
                imgsz=int(self.model_trainer_config.image_size),
                batch=int(self.model_trainer_config.batch),
                project=self.model_trainer_config.artifact_dir,
                name=".",
                exist_ok=True,
                conf=self.model_trainer_config.conf_threshold,
                iou=self.model_trainer_config.iou_threshold,
                verbose=True,
            )

            # Expected paths (Ultralytics convention under artifact_dir/weights)
            best_path = Path(self.model_trainer_config.best_weights_path)
            last_path = Path(self.model_trainer_config.last_weights_path)

            if not best_path.exists():
                logging.warning("Expected best.pt not found at %s", str(best_path))

            artifact = ODModelTrainerArtifact(
                best_weights_path=str(best_path),
                last_weights_path=str(last_path) if last_path.exists() else None,
                training_artifact_dir=self.model_trainer_config.artifact_dir,
                mlflow_run_id=mlflow_run_id,
            )

            logging.info("Exited initiate_model_trainer (YOLO)")
            return artifact
        except Exception as e:
            raise XRayException(e, sys)