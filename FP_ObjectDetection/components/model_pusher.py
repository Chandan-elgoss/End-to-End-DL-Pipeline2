import sys
from pathlib import Path
import shutil

from FP_ObjectDetection.entity.artifact_entity import ModelPusherArtifact, ODModelTrainerArtifact
from FP_ObjectDetection.entity.config_entity import ModelPusherConfig
from FP_ObjectDetection.exception import XRayException
from FP_ObjectDetection.logger import logging


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_trainer_artifact: ODModelTrainerArtifact):
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifact = model_trainer_artifact

    # NOTE (legacy): This repository previously built and pushed a BentoML image to ECR.
    # That flow is intentionally disabled for this local-only object-detection pipeline.
    # def build_and_push_bento_image(self):
    #     ...

    def deploy_locally(self) -> Path:
        """Copy best weights to a stable local deployment path."""
        best_path = Path(self.model_trainer_artifact.best_weights_path)
        if not best_path.exists():
            raise FileNotFoundError(f"Best weights not found: {best_path}")

        deployed_dir = Path(self.model_pusher_config.deployed_models_dir)
        deployed_dir.mkdir(parents=True, exist_ok=True)
        target = deployed_dir / self.model_pusher_config.deployed_model_filename
        shutil.copy2(best_path, target)
        return target
        


    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.

        Output      :   Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            pushed_path = self.deploy_locally()

            model_pusher_artifact = ModelPusherArtifact(
                pushed_model_path=str(pushed_path),
                deployed_models_dir=self.model_pusher_config.deployed_models_dir,
            )

            logging.info("Exited the initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise XRayException(e, sys)
