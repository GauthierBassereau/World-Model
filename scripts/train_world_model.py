import pyrallis
from training.world_trainer import (
    WorldModelTrainer,
    WorldModelTrainingConfig,
)
from world_model.backbone import WorldModelBackbone

# Use it like:
# python scripts/train_world_model.py configs/pretraining.yaml

def main() -> None:
    config = pyrallis.parse(config_class=WorldModelTrainingConfig)
    
    model = WorldModelBackbone(config.world_model)
    trainer = WorldModelTrainer(config, model)
    trainer.train()


if __name__ == "__main__":
    main()
