from ..base.base_server import BaseServer
from utils.data_utils import get_dataset_loader


class ByzantineBaseServer(BaseServer):
    "Abstract Server class for Byzantine-resilient based methods. It adds trust dataset"

    def __init__(self, cfg, trust_df=None):
        super().__init__(cfg)
        self.trust_df = trust_df
        if self.trust_df is not None:
            self.trust_loader = get_dataset_loader(
                self.trust_df, self.cfg, drop_last=False
            )
