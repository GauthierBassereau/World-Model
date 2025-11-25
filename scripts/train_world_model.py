import pyrallis
from src.training.world_trainer import (
    WorldModelTrainer,
    WorldModelTrainingConfig,
)
from src.world_model.backbone import WorldModelBackbone

def main() -> None:
    config = pyrallis.parse(config_class=WorldModelTrainingConfig)
    
    model = WorldModelBackbone(config.world_model)
    trainer = WorldModelTrainer(config, model)
    trainer.train()


if __name__ == "__main__":
    main()
