import sys
import copy
import time
from collections import OrderedDict
from utils.data_utils import get_dataset_loader
from hydra.utils import instantiate

from utils.losses import get_loss
from utils.attack_utils import add_attack_functionality
from utils.process_utils import errors_child_handler


class FedAvgClient:
    def __init__(self, *client_args, **client_kwargs):
        self._original_cls = type(self)
        self.client_args = client_args
        self.client_kwargs = client_kwargs
        cfg = self.client_args[0]
        dataset = self.client_args[1]
        self.cfg = cfg
        self.global_dataset = dataset
        self.rank = client_kwargs["rank"]
        self.pipe = client_kwargs["pipe"]
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.criterion = None
        self.server_model_state = None
        self.print_metrics = cfg.federated_params.print_client_metrics
        self.train_val_prop = cfg.federated_params.client_train_val_prop
        self.device = (
            "{}:{}".format(
                cfg.training_params.device,
                cfg.training_params.device_ids[
                    self.rank % len(cfg.training_params.device_ids)
                ],
            )
            if cfg.training_params.device == "cuda"
            else "cpu"
        )

        if "model" in self.client_kwargs:
            assert "model_trainer" in self.client_kwargs
            self.model = self.client_kwargs.pop("model")
            self.model_trainer = self.client_kwargs.pop("model_trainer")
        else:
            self.model = instantiate(
                cfg.model, num_classes=self.global_dataset.num_classes
            )
            # Instantiate model_trainer which will be responsible for technical training of model
            self.model_trainer = instantiate(
                self.cfg.model_trainer, cfg=self.cfg, _recursive_=False
            )

        self.model.to(self.device)
        self._set_client_dataset()
        self._init_loaders()
        self._init_optimizer()
        self._init_criterion()
        self.pipe_commands_map = self.create_pipe_commands()

        self.grad = OrderedDict()
        self.local_epochs = self.cfg.federated_params.local_epochs

    def _init_optimizer(self):
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())

    def _init_criterion(self):
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.train_dataset.data,
            num_classes=self.cfg.training_params.num_classes,
        )

    def _set_client_dataset(self):
        self.train_dataset = self.global_dataset.to_client_side(self.rank)

    def _init_loaders(self):
        self.valid_dataset = self.train_dataset.dataset_split(self.train_val_prop)

        self.train_loader = get_dataset_loader(
            self.train_dataset, self.cfg, drop_last=False
        )
        self.valid_loader = get_dataset_loader(
            self.valid_dataset, self.cfg, drop_last=False
        )

    def _set_attack_type(self, attack_content):
        self.attack_type = attack_content[0]
        self.attack_config = attack_content[1]

    def reinit_self(self, new_rank):
        # new_kwargs = dict(self.client_kwargs)
        args = self.client_args
        kwargs = self.client_kwargs
        kwargs["rank"] = new_rank
        kwargs["model"] = self.model
        kwargs["model_trainer"] = self.model_trainer
        original_cls = self._original_cls
        self.__dict__.clear()
        original_cls.__init__(self, *args, **kwargs)

        # Recive content for local learning
        content = self.pipe.recv()
        self.parse_communication_content(content)

    def shutdown_self(self):
        print(f"Exit child {self.rank} process")
        sys.exit(0)

    def create_pipe_commands(self):
        # define a structure to process pipe values
        pipe_commands_map = {
            "update_model": lambda state_dict: self.model.load_state_dict(
                {k: v.to(self.device) for k, v in state_dict.items()}
            ),
            "attack_type": self._set_attack_type,
            "shutdown": lambda _: self.shutdown_self(),
            "reinit": lambda new_rank: self.reinit_self(new_rank),
        }

        return pipe_commands_map

    def get_loss_value(self, outputs, targets):
        return self.criterion(outputs, targets)

    def get_grad(self):
        self.model.eval()
        for key, _ in self.model.state_dict().items():
            self.grad[key] = (
                self.model.state_dict()[key] - self.server_model_state[key]
            ).to("cpu")

    def train(self):
        start = time.time()

        # Save the server model state to get_grad
        self.server_model_state = copy.deepcopy(self.model).state_dict()

        # Validate server weights before training to set up best model
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )

        # Train client
        self.model_trainer.train_fn(self)

        # Get client metrics
        if self.print_metrics:
            self.client_val_loss, self.client_metrics = (
                self.model_trainer.client_eval_fn(self)
            )

        # Calculate client update
        self.get_grad()

        # Save training time
        self.result_time = time.time() - start

    def get_communication_content(self):
        # In fedavg_client we need to send only result of local learning
        result_dict = {
            "grad": self.grad,
            "rank": self.rank,
            "time": self.result_time,
            "server_metrics": (
                self.server_metrics,
                self.server_val_loss,
                len(self.valid_dataset),
            ),
        }
        if self.print_metrics:
            result_dict["client_metrics"] = (self.client_val_loss, self.client_metrics)

        return result_dict

    def parse_communication_content(self, content):
        # In fedavg_client we need to recive model after aggregate and
        # attack type for this client
        for key, value in content.items():
            if key in self.pipe_commands_map.keys():
                self.pipe_commands_map[key](copy.deepcopy(value))
            else:
                raise ValueError(
                    f"Recieved content in client {self.rank} from server, with unknown key={key}"
                )


@errors_child_handler
def multiprocess_client(*client_args, client_cls, pipe, rank, attack_type):
    # Init client instance
    client_kwargs = {"pipe": pipe, "rank": rank}
    client = client_cls(*client_args, **client_kwargs)

    # Loop of federated learning
    while True:
        # Wait content from server to start local learning
        content = client.pipe.recv()
        client.parse_communication_content(content)

        # Can be this realization of attack
        if client.attack_type != "no_attack":
            client = add_attack_functionality(
                client, client.attack_type, client.attack_config
            )

        client.train()

        # Send content to server, local learning ended
        content = client.get_communication_content()
        client.pipe.send(content)
