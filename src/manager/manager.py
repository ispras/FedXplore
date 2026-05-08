from hydra.utils import instantiate
import torch.multiprocessing as mp
from federated_methods.fedavg.fedavg_client import multiprocess_client


class Manager:
    def __init__(self, cfg, server, df, batch_generator, **kwargs) -> None:
        self.server = server
        self.cfg = cfg
        self.amount_of_clients = self.cfg.federated_params.amount_of_clients
        self.batch_generator = instantiate(
            batch_generator,
            amount_of_clients=self.amount_of_clients,
            df=df,
        )

    def set_ranks_to_procs(self, clients_batch):
        # Step the manager to update ranks for clients
        for client_idx, new_rank in enumerate(clients_batch):
            content = {"reinit": new_rank}
            self.server.send_content_to_client(client_idx, content)

    def create_batches(self, list_clients):
        self.batch_generator.create_batches(list_clients)
        return self.batch_generator.batches

    def create_clients(self, client_args, client_kwargs, attack_map):
        self.processes = []

        # Init pipe for every client
        self.pipes = [mp.Pipe() for _ in range(self.batch_generator.batch_size)]
        self.server.pipes = [pipe[0] for pipe in self.pipes]  # Init input (server) pipe

        for rank in range(self.batch_generator.batch_size):
            # Every process starts by calling the same function with the given arguments
            client_kwargs["pipe"] = self.pipes[rank][1]  # Send current pipe
            client_kwargs["rank"] = rank
            client_kwargs["attack_type"] = attack_map[rank]
            p = mp.Process(
                target=multiprocess_client,
                args=client_args,
                kwargs=client_kwargs,
            )
            p.start()
            self.processes.append(p)

    def stop_train(self):
        # Close all client proccesses
        for client_idx in range(self.batch_generator.batch_size):
            content = {"shutdown": None}
            self.server.send_content_to_client(client_idx, content)

        for p in self.processes:
            p.join()

        exit(0)
