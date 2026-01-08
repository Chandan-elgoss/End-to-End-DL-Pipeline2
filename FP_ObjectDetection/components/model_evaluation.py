import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from ultralytics import YOLO

from FP_ObjectDetection.entity.artifact_entity import ODModelEvaluationArtifact, ODModelTrainerArtifact
from FP_ObjectDetection.entity.config_entity import ModelEvaluationConfig
from FP_ObjectDetection.exception import XRayException
from FP_ObjectDetection.logger import logging


class ModelEvaluation:
    """YOLO evaluation + MLflow logging.

    Runs `model.val()` and logs key metrics + best weights as an MLflow artifact.
    """

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ODModelTrainerArtifact,
    ):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact

    def _start_or_resume_mlflow(self) -> Optional[str]:
        mlflow.set_tracking_uri(self.model_evaluation_config.mlflow_tracking_uri)
        mlflow.set_experiment(self.model_evaluation_config.mlflow_experiment_name)
        if self.model_trainer_artifact.mlflow_run_id:
            mlflow.start_run(run_id=self.model_trainer_artifact.mlflow_run_id)
            return self.model_trainer_artifact.mlflow_run_id
        run = mlflow.start_run(run_name=self.model_evaluation_config.mlflow_run_name or None)
        return run.info.run_id

    @staticmethod
    def _coerce_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
        coerced: Dict[str, float] = {}
        for k, v in metrics.items():
            try:
                coerced[k] = float(v)
            except Exception:
                continue
        return coerced

    def initiate_model_evaluation(self) -> ODModelEvaluationArtifact:
        logging.info("Entered initiate_model_evaluation (YOLO)")
        try:
            os_best = Path(self.model_trainer_artifact.best_weights_path)
            if not os_best.exists():
                raise FileNotFoundError(
                    f"Trained best weights not found: {self.model_trainer_artifact.best_weights_path}"
                )

            model = YOLO(str(os_best))
            val_results = model.val(
                data=self.model_evaluation_config.dataset_yaml_path,
                conf=self.model_evaluation_config.conf_threshold,
                iou=self.model_evaluation_config.iou_threshold,
                verbose=False,
            )

            raw_metrics: Dict[str, Any] = {}
            # Ultralytics exposes metrics differently across versions.
            if hasattr(val_results, "results_dict"):
                raw_metrics = dict(val_results.results_dict)
            elif hasattr(val_results, "metrics"):
                raw_metrics = dict(getattr(val_results, "metrics"))

            metrics = self._coerce_metrics(raw_metrics)

            run_id = self._start_or_resume_mlflow()
            try:
                if metrics:
                    mlflow.log_metrics(metrics)
                mlflow.log_params(
                    {
                        "dataset_yaml": self.model_evaluation_config.dataset_yaml_path,
                        "conf": self.model_evaluation_config.conf_threshold,
                        "iou": self.model_evaluation_config.iou_threshold,
                    }
                )
                mlflow.log_artifact(str(os_best), artifact_path="model")
            finally:
                mlflow.end_run()

            artifact = ODModelEvaluationArtifact(metrics=metrics, mlflow_run_id=run_id)
            logging.info("Exited initiate_model_evaluation (YOLO)")
            return artifact
        except Exception as e:
            raise XRayException(e, sys)
