from __future__ import annotations

from flwr.app import ArrayRecord, ConfigRecord, Message
from flwr.serverapp import Grid

from strategies.fedavg import PhysicalFedAvg


class PhysicalFedProx(PhysicalFedAvg):
    def __init__(self, *args, proximal_mu: float = 0.01, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proximal_mu = float(proximal_mu)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        cfg = ConfigRecord(dict(config))
        cfg["proximal-mu"] = self.proximal_mu
        return super().configure_train(server_round, arrays, cfg, grid)