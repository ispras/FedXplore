import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.process_utils import errors_parent_handler


@hydra.main(version_base=None, config_path="../configs", config_name="config")
@errors_parent_handler
def train(cfg: DictConfig):

    # Init federated_method
    trainer = instantiate(cfg.federated_method, _recursive_=False)
    trainer._init_federated(cfg)

    # Add client selection strategy
    client_selector = instantiate(
        trainer.cfg.client_selector, cfg=trainer.cfg, _recursive_=False
    )
    trainer = client_selector(trainer)

    trainer.begin_train()


if __name__ == "__main__":
    train()
