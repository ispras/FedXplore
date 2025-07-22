import hydra
import signal
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.utils import handle_main_process_sigterm
from utils.logging_utils import redirect_stdout_to_log


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    redirect_stdout_to_log()

    # Init train dataset
    df = instantiate(cfg.train_dataset, cfg=cfg, mode="train", _recursive_=False)
    cfg = df.get_cfg()

    # Init federated_method
    trainer = instantiate(cfg.federated_method, _recursive_=False)
    trainer._init_federated(cfg, df)

    # Add client selection strategy
    client_selector = instantiate(cfg.client_selector, cfg=cfg, _recursive_=False)
    trainer = client_selector(trainer)

    # Termination handling in multiprocess setup
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: handle_main_process_sigterm(signum, frame, trainer),
    )

    trainer.begin_train()


if __name__ == "__main__":
    train()
