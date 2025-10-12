import hydra
from trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg) -> None:
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
