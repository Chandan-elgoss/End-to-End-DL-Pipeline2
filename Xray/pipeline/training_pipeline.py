import sys
from Xray.components.model_training import ModelTrainer
from Xray.components.model_evaluation import ModelEvaluation
from Xray.components.model_pusher import ModelPusher
from Xray.exception import XRayException
from Xray.logger import logging
from Xray.entity.artifact_entity import (
    ODModelTrainerArtifact,
    ODModelEvaluationArtifact,
    ModelPusherArtifact,
    )

from Xray.entity.config_entity import (
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

class TrainPipeline:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config=ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        
        
    # NOTE (legacy): Data ingestion / transformation were for the Lung X-ray
    # classifier and are intentionally disabled for the YOLO object detection
    # pipeline.
    #
    # def start_data_ingestion(...):
    # def start_data_transformation(...):
    def start_model_trainer(self) -> ODModelTrainerArtifact:
        logging.info("Entered the start_model_trainer method of TrainPipeline class")
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.model_trainer_config)
            trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            return trainer_artifact
        except Exception as e:
            raise XRayException(e, sys)
    
    def start_model_evaluation(
        self,
        model_trainer_artifact: ODModelTrainerArtifact,
    ) -> ODModelEvaluationArtifact:
        logging.info("Entered the start_model_evaluation method of TrainPipeline class")

        try:
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifact=model_trainer_artifact,
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            logging.info(
                "Exited the start_model_evaluation method of TrainPipeline class"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)

    def start_model_pusher(self, model_trainer_artifact: ODModelTrainerArtifact) -> ModelPusherArtifact:
        logging.info("Entered the start_model_pusher method of TrainPipeline class")

        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifact=model_trainer_artifact,
            )

            model_pusher_artifact = model_pusher.initiate_model_pusher()

            logging.info("Exited the start_model_pusher method of TrainPipeline class")

            return model_pusher_artifact

        except Exception as e:
            raise XRayException(e, sys)

    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")

        try:
            model_trainer_artifact = self.start_model_trainer()
            _ = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact)
            _ = self.start_model_pusher(model_trainer_artifact=model_trainer_artifact)

            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise XRayException(e, sys)

            
            
        
        