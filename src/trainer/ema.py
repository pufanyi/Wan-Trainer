"""Exponential Moving Average for FSDP2 training."""

import torch


class EMA:
    """Exponential Moving Average of model parameters.

    Works with FSDP2: each rank maintains EMA of its local parameter shards.
    Does not support resharding (changing world_size on resume).
    """

    def __init__(self, params: list[torch.nn.Parameter], decay: float):
        self.decay = decay
        self.params = params
        self.shadow = [p.data.clone() for p in params]

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params, strict=True):
            s.lerp_(p.data, 1 - self.decay)

    def state_dict(self) -> list[torch.Tensor]:
        return [s.clone() for s in self.shadow]

    def load_state_dict(self, state: list[torch.Tensor]):
        for s, loaded in zip(self.shadow, state, strict=True):
            s.copy_(loaded)
