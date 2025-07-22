from hydra.utils import instantiate
import torch.multiprocessing as mp
from federated_methods.base.base_client import multiprocess_client


class Manager:
    def __init__(self, cfg, server) -> None:
        self.server = server
        self.cfg = cfg
        self.amount_of_clients = self.cfg.federated_params.amount_of_clients
        self.batches = instantiate(
            cfg.manager, amount_of_clients=self.amount_of_clients
        )

    def step(self, batch_idx):
        # Step the manager to update ranks for clients
        next_batch = self.batches.get_batch((batch_idx + 1) % len(self.batches))
        for client_idx, new_rank in enumerate(next_batch):
            content = {"reinit": new_rank}
            self.server.send_content_to_client(client_idx, content)

    def get_clients_loader(self):
        return self.batches

    def create_clients(self, client_args, client_kwargs, attack_map):
        self.processes = []

        # Init pipe for every client
        self.pipes = [mp.Pipe() for _ in range(self.batches.batch_size)]
        self.server.pipes = [pipe[0] for pipe in self.pipes]  # Init input (server) pipe

        for rank in range(self.batches.batch_size):
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
        for client_idx in range(self.batches.batch_size):
            content = {"shutdown": None}
            self.server.send_content_to_client(client_idx, content)
            
        for p in self.processes:
            p.join()

        exit(0)


class SequentialIterator:
    def __init__(self, batch_size, amount_of_clients):
        self.amount_of_clients = amount_of_clients
        self.ranks = [i for i in range(self.amount_of_clients)]
        self.batch_size = self.define_batch_len(batch_size)
        self.num_batches = len(self.ranks) // batch_size + (
            1 if len(self.ranks) % batch_size != 0 else 0
        )

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx < len(self.ranks):
            batch = self.ranks[self.current_idx : self.current_idx + self.batch_size]
            self.current_idx += self.batch_size
            return batch
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def define_batch_len(self, batch_size):
        if batch_size == "dynamic":
            # IMPLEMENT LATER
            assert (
                False
            ), "At the current moment we do not support dynamic size of processes batch"
        else:
            return min(self.amount_of_clients, batch_size)

    def get_batch(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.ranks[start:end]
    def get_name(self):
        return "Sequential"


class ClientSelectionIterator(SequentialIterator):
    def __init__(self, batch_size, amount_of_clients):
        self.amount_of_clients = amount_of_clients
        self.batch_size = self.define_batch_len(batch_size)

        self.num_batches = None
        self.current_batch = None

    def create_batch(self, list_of_clients):
        self.current_batch = [
            list_of_clients[i : i + self.batch_size]
            for i in range(0, len(list_of_clients), self.batch_size)
        ]
        self.num_batches = len(self.current_batch)

    def __next__(self):
        if self.current_idx < self.num_batches:
            self.current_idx += 1
            return self.current_batch[self.current_idx - 1]
        else:
            raise StopIteration

    def get_batch(self, idx):
        if idx == -1:
            # Reinit on the required clients before the start of the round
            # The first batch is not important
            return self.current_batch[0]

        return self.current_batch[idx]

    def get_name(self):
        return "ClientSelection"
