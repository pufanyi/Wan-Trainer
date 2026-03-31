"""DCP (Distributed Checkpoint) state management for FSDP2 training."""

import torch
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful


class TrainState(Stateful):
    """Wraps model + optimizer for DCP save/load via the Stateful protocol.

    DCP calls state_dict() / load_state_dict() automatically.
    get_state_dict / set_state_dict handle FSDP2 FQN translation and sharding.

    Each transformer has its own optimizer so that get_state_dict can correctly
    map parameter IDs to FQNs.
    """

    def __init__(
        self,
        transformer: torch.nn.Module | None = None,
        transformer_2: torch.nn.Module | None = None,
        optimizer_1: torch.optim.Optimizer | None = None,
        optimizer_2: torch.optim.Optimizer | None = None,
        step: int = 0,
        epoch: int = 0,
        batch_idx: int = 0,
    ):
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2
        self.step = step
        self.epoch = epoch
        self.batch_idx = batch_idx

    def state_dict(self):
        sd = {"step": self.step, "epoch": self.epoch, "batch_idx": self.batch_idx}
        if self.transformer is not None and self.optimizer_1 is not None:
            t1_model_sd, t1_optim_sd = get_state_dict(self.transformer, self.optimizer_1)
            sd["transformer"] = t1_model_sd
            sd["optimizer_transformer"] = t1_optim_sd
        if self.transformer_2 is not None and self.optimizer_2 is not None:
            t2_model_sd, t2_optim_sd = get_state_dict(self.transformer_2, self.optimizer_2)
            sd["transformer_2"] = t2_model_sd
            sd["optimizer_transformer_2"] = t2_optim_sd
        return sd

    def load_state_dict(self, state_dict):
        if self.transformer is not None and self.optimizer_1 is not None and "transformer" in state_dict:
            set_state_dict(
                self.transformer,
                self.optimizer_1,
                model_state_dict=state_dict["transformer"],
                optim_state_dict=state_dict["optimizer_transformer"],
            )
        if self.transformer_2 is not None and self.optimizer_2 is not None and "transformer_2" in state_dict:
            set_state_dict(
                self.transformer_2,
                self.optimizer_2,
                model_state_dict=state_dict["transformer_2"],
                optim_state_dict=state_dict["optimizer_transformer_2"],
            )
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.batch_idx = state_dict.get("batch_idx", 0)
