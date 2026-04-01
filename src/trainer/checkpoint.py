"""DCP (Distributed Checkpoint) state management for FSDP2 training."""

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful


def _save_pair(sd: dict, key: str, model, optimizer):
    """Save model + optimizer state_dict pair into sd[key] and sd[key_optim]."""
    if model is not None and optimizer is not None:
        m_sd, o_sd = get_state_dict(model, optimizer)
        sd[key] = m_sd
        sd[f"optimizer_{key}"] = o_sd


def _load_pair(state_dict: dict, key: str, model, optimizer):
    """Load model + optimizer state_dict pair from state_dict."""
    if model is not None and optimizer is not None and key in state_dict:
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict[key],
            optim_state_dict=state_dict[f"optimizer_{key}"],
        )


class TrainState(Stateful):
    """Wraps models + optimizers + RNG states for DCP save/load via the Stateful protocol.

    Each FSDP-sharded module has its own optimizer so that get_state_dict can
    correctly map parameter IDs to FQNs.
    """

    def __init__(
        self,
        text_encoder: torch.nn.Module | None = None,
        transformer: torch.nn.Module | None = None,
        transformer_2: torch.nn.Module | None = None,
        optimizer_te: torch.optim.Optimizer | None = None,
        optimizer_1: torch.optim.Optimizer | None = None,
        optimizer_2: torch.optim.Optimizer | None = None,
        step: int = 0,
        epoch: int = 0,
        batch_idx: int = 0,
    ):
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.optimizer_te = optimizer_te
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2
        self.step = step
        self.epoch = epoch
        self.batch_idx = batch_idx

    def state_dict(self):
        sd = {"step": self.step, "epoch": self.epoch, "batch_idx": self.batch_idx}
        _save_pair(sd, "text_encoder", self.text_encoder, self.optimizer_te)
        _save_pair(sd, "transformer", self.transformer, self.optimizer_1)
        _save_pair(sd, "transformer_2", self.transformer_2, self.optimizer_2)
        # RNG states for reproducibility on resume
        sd["rng_cpu"] = torch.random.get_rng_state()
        sd["rng_cuda"] = torch.cuda.get_rng_state()
        return sd

    def load_state_dict(self, state_dict):
        _load_pair(state_dict, "text_encoder", self.text_encoder, self.optimizer_te)
        _load_pair(state_dict, "transformer", self.transformer, self.optimizer_1)
        _load_pair(state_dict, "transformer_2", self.transformer_2, self.optimizer_2)
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.batch_idx = state_dict.get("batch_idx", 0)
        # Restore RNG states
        if "rng_cpu" in state_dict:
            torch.random.set_rng_state(state_dict["rng_cpu"])
        if "rng_cuda" in state_dict:
            torch.cuda.set_rng_state(state_dict["rng_cuda"])


def load_dcp_into_pipeline(pipe, checkpoint_path: str, use_ema: bool = False) -> None:
    """Load a DCP training checkpoint into a diffusers pipeline for inference.

    Handles resharding automatically: a checkpoint saved on N GPUs with FSDP2
    can be loaded on a single GPU for inference.

    Args:
        pipe: A WanImageToVideoPipeline (or similar) with .transformer / .transformer_2.
        checkpoint_path: Path to the DCP checkpoint directory (contains .metadata).
        use_ema: If True, load EMA shadow weights instead of model weights.
    """
    needs_cleanup = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")
        needs_cleanup = True

    try:
        transformers = {}
        if getattr(pipe, "transformer", None) is not None:
            transformers["transformer"] = pipe.transformer
        if getattr(pipe, "transformer_2", None) is not None:
            transformers["transformer_2"] = pipe.transformer_2

        if use_ema:
            _load_ema_into_models(transformers, checkpoint_path)
        else:
            _load_train_state_into_models(transformers, checkpoint_path)
    finally:
        if needs_cleanup:
            dist.destroy_process_group()


def _load_train_state_into_models(
    transformers: dict[str, torch.nn.Module],
    checkpoint_path: str,
) -> None:
    """Load model weights from train_state in a DCP checkpoint."""
    state: dict = {"train_state": {}}
    for name, model in transformers.items():
        state["train_state"][name] = model.state_dict()

    dcp.load(state, checkpoint_id=checkpoint_path)

    for name, model in transformers.items():
        model.load_state_dict(state["train_state"][name])
    logger.info("Loaded model weights from DCP checkpoint {}", checkpoint_path)


def _load_ema_into_models(
    transformers: dict[str, torch.nn.Module],
    checkpoint_path: str,
) -> None:
    """Load EMA shadow weights from a DCP checkpoint into models."""
    shadow: dict[str, torch.Tensor] = {}
    for model_name, model in transformers.items():
        for pname, p in model.named_parameters():
            shadow[f"{model_name}.{pname}"] = torch.empty_like(p, device="cpu")

    state: dict = {"ema": {"shadow": shadow, "decay": torch.tensor(0.0)}}
    dcp.load(state, checkpoint_id=checkpoint_path)

    for model_name, model in transformers.items():
        prefix = f"{model_name}."
        sd = {k.removeprefix(prefix): v for k, v in shadow.items() if k.startswith(prefix)}
        model.load_state_dict(sd)
    logger.info("Loaded EMA weights from DCP checkpoint {}", checkpoint_path)
