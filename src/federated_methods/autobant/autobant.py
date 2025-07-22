from ..byzantine_base.byzantine import ByzantineBase
from .autobant_server import AutoBANTServer
from hydra.utils import instantiate


class AutoBANT(ByzantineBase):
    def __init__(
        self,
        start_point,
        end_point,
        num_opt_epochs,
        mirror_gamma,
        ts_momentum,
    ):
        super().__init__()
        self.opt_params = [
            start_point,
            end_point,
            num_opt_epochs,
            mirror_gamma,
            ts_momentum,
        ]

    def _init_server(self, cfg):
        super()._init_server(cfg)
        self.server = AutoBANTServer(cfg, self.trust_df, *self.opt_params)

    def count_trust_scores(self):
        return self.server._count_trust_score()
