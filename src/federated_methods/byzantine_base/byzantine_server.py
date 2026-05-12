from ..fedavg.fedavg_server import FedAvgServer
from utils.data_utils import get_dataset_loader


class ByzantineBaseServer(FedAvgServer):
    "Abstract Server class for Byzantine-resilient based methods. It adds trust dataset"

    def __init__(self, cfg, trust_df=None):
        super().__init__(cfg)
        self.trust_df = trust_df
        if self.trust_df is not None:
            self.trust_loader = get_dataset_loader(
                self.trust_df, self.cfg, drop_last=False
            )
