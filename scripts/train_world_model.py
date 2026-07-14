import pyrallis
from src.training.world_trainer import (
    WorldModelTrainer,
    WorldModelTrainingConfig,
)
from src.world_model.backbone import WorldModelBackbone
from src.rae_dino import (
    build_autoencoder,
    configured_autoencoder_resolution,
    validate_autoencoder_input_dim,
)

def main() -> None:
    config = pyrallis.parse(config_class=WorldModelTrainingConfig)
    image_size = configured_autoencoder_resolution(config.autoencoder)
    config.train_dataset.image_size = image_size
    config.eval_dataset.image_size = image_size
    
    model = WorldModelBackbone(config.world_model)
    autoencoder = build_autoencoder(config.autoencoder)
    validate_autoencoder_input_dim(autoencoder, config.world_model.input_dim)
    trainer = WorldModelTrainer(config, model, autoencoder)
    trainer.train()


if __name__ == "__main__":
    main()
