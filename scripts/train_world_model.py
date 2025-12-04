import pyrallis
from src.training.world_trainer import (
    WorldModelTrainer,
    WorldModelTrainingConfig,
)
from src.world_model.backbone import WorldModelBackbone
from src.rae_dino.rae import RAE

def main() -> None:
    config = pyrallis.parse(config_class=WorldModelTrainingConfig)
    
    model = WorldModelBackbone(config.world_model)
    autoencoder = RAE()
    trainer = WorldModelTrainer(config, model, autoencoder)
    trainer.train()


if __name__ == "__main__":
    main()
