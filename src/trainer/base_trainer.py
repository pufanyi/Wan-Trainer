"""Base trainer with shared infrastructure for FSDP2 + DCP training."""

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from src.data.i2v_dataset import I2VDataset
from src.models.wan_i2v import LoRATrainConfig, WanI2VForTraining
from src.trainer.checkpoint import TrainState
from src.trainer.config import TrainConfig
from src.trainer.ema import EMA
from src.trainer.utils import apply_liger_rms_norm, collate, cosine_lr, setup_loguru, shard_transformer


class BaseTrainer:
    """Shared infrastructure: distributed init, model, FSDP2, EMA, compile,
    dataset, optimizer, DCP checkpointing, wandb, and resume logic.

    Subclasses implement ``train()`` and any training-mode-specific setup.
    Override hooks:
        ``_pre_fsdp_setup``  — called after model build, before FSDP sharding.
        ``_setup_fsdp``      — override to shard additional modules (call super).
        ``_post_init``       — called after all base init, before resume.
        ``_compute_total_steps`` — controls total optimizer steps.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        # ---- Distributed ----
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(cfg.seed + self.rank)

        # ---- Expert parallel (must run before model build) ----
        self._init_expert_parallel(cfg)

        setup_loguru(self.rank)
        logger.info("World size: {}", self.world_size)

        # ---- Model ----
        self.model = self._build_model(cfg)

        # ---- Subclass hook (e.g. create reference policy copies) ----
        self._pre_fsdp_setup(cfg)

        # ---- FSDP2 ----
        self.mesh, self.mp_policy = self._create_device_mesh(cfg)
        self.sync_modules = self._setup_fsdp(cfg)

        # ---- EMA ----
        self.ema = self._setup_ema(cfg)

        # ---- torch.compile ----
        if cfg.torch_compile:
            self._compile_modules(cfg)

        # ---- Dataset / DataLoader ----
        self.dataset, self.sampler = self._build_dataset(cfg)
        self.dataloader = self._build_dataloader(self.dataset, cfg)

        # ---- Optimizer ----
        self.params, self.optimizers, self.optimizer_te, self.optimizer_1, self.optimizer_2 = (
            self._build_optimizers(cfg)
        )

        # ---- Total steps ----
        self.total_steps = self._compute_total_steps()
        logger.info(
            "Dataset: {} samples, {} batches/epoch, {} total optimizer steps",
            len(self.dataset), len(self.dataloader), self.total_steps,
        )

        # ---- DCP state ----
        self.train_state = TrainState(
            text_encoder=self.model.text_encoder if cfg.train_text_encoder else None,
            transformer=self.model.transformer,
            transformer_2=self.model.transformer_2,
            optimizer_te=self.optimizer_te,
            optimizer_1=self.optimizer_1,
            optimizer_2=self.optimizer_2,
        )

        # ---- Subclass hook (e.g. MFU monitor) ----
        self._post_init(cfg)

        # ---- Wandb ----
        self.use_wandb = cfg.wandb_project is not None and self.rank == 0
        if self.use_wandb:
            import wandb
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg.model_dump())

        # ---- Resume ----
        auto_resume_path = self._find_latest_checkpoint() if cfg.auto_resume else None
        resume_path = auto_resume_path or cfg.resume_from
        if resume_path:
            is_auto_resume = auto_resume_path is not None
            if cfg.reset_dataloader is None:
                self._reset_on_load = not is_auto_resume
            else:
                self._reset_on_load = cfg.reset_dataloader
            self._load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _pre_fsdp_setup(self, cfg: TrainConfig) -> None:
        """Called after model build, before FSDP. Override to create ref models etc."""

    def _init_expert_parallel(self, cfg: TrainConfig) -> None:
        """Set up expert-parallel state: split GPUs into two groups, one per MoE expert."""
        self.expert_parallel = cfg.expert_parallel
        if not self.expert_parallel:
            self._effective_train_experts = cfg.train_experts
            self.expert_group = -1
            self.dp_rank = self.rank
            self.dp_size = self.world_size
            return

        assert cfg.train_experts == "both", (
            f"expert_parallel requires train_experts='both', got '{cfg.train_experts}'"
        )
        assert self.world_size >= 2 and self.world_size % 2 == 0, (
            f"expert_parallel requires even world_size >= 2, got {self.world_size}"
        )

        half = self.world_size // 2
        self.expert_group = 0 if self.rank < half else 1
        self.dp_rank = self.rank % half
        self.dp_size = half
        self._effective_train_experts = "high" if self.expert_group == 0 else "low"

    def _get_expert_parallel_sampler_seed(self, cfg: TrainConfig) -> int:
        """Sampler seed for expert-parallel mode.

        Default: same seed -> both groups iterate the same data (SFT behavior).
        Override in subclass for per-group independent data (COS behavior).
        """
        return cfg.seed

    def _post_init(self, cfg: TrainConfig) -> None:
        """Called after base init, before wandb/resume. Override for MFU etc."""

    def _compute_total_steps(self) -> int:
        """Total optimizer steps. Override for different accumulation strategies."""
        return self.cfg.num_epochs * len(self.dataloader) // self.cfg.gradient_accumulation_steps

    def train(self):
        """Main training loop. Must be implemented by subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _build_model(self, cfg: TrainConfig) -> WanI2VForTraining:
        train_experts = self._effective_train_experts
        lora_cfg = (
            LoRATrainConfig(rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
            if cfg.lora_rank > 0
            else None
        )
        logger.info(
            "Loading model from {} (lora_rank={}, experts={}{}) ...",
            cfg.model_path, cfg.lora_rank, train_experts,
            f", expert_group={self.expert_group}" if self.expert_parallel else "",
        )
        model = WanI2VForTraining(
            cfg.model_path,
            lora_config=lora_cfg,
            train_experts=train_experts,
            train_text_encoder=cfg.train_text_encoder,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )
        if cfg.use_liger_kernel:
            count = 0
            for m in [model.transformer, model.transformer_2]:
                if m is not None:
                    count += apply_liger_rms_norm(m)
            logger.info("Liger Kernel: replaced {} RMSNorm modules", count)
        model.text_encoder.to(self.device)
        model.vae.to(self.device)
        return model

    # ------------------------------------------------------------------
    # FSDP2
    # ------------------------------------------------------------------

    def _create_device_mesh(self, cfg: TrainConfig):
        if self.expert_parallel:
            mesh_2d = init_device_mesh(
                "cuda", (2, self.dp_size), mesh_dim_names=("expert", "dp"),
            )
            mesh = mesh_2d["dp"]
            self._dp_pg = mesh.get_group()
            logger.info(
                "Expert parallel mesh: group={} dp_rank={}/{}",
                self.expert_group, self.dp_rank, self.dp_size,
            )
        else:
            mesh = init_device_mesh("cuda", (self.world_size,))
            self._dp_pg = None

        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype_map[cfg.param_dtype],
            reduce_dtype=dtype_map[cfg.reduce_dtype],
        )
        return mesh, mp_policy

    def _setup_fsdp(self, cfg: TrainConfig) -> list[torch.nn.Module]:
        """Shard trainable modules with FSDP2. Override to shard additional modules."""
        if cfg.train_text_encoder:
            fully_shard(self.model.text_encoder, mesh=self.mesh, mp_policy=self.mp_policy)
        if self.model.transformer is not None:
            shard_transformer(self.model.transformer, self.mesh, self.mp_policy)
        if self.model.transformer_2 is not None:
            shard_transformer(self.model.transformer_2, self.mesh, self.mp_policy)
        return [
            m for m in [
                self.model.text_encoder if cfg.train_text_encoder else None,
                self.model.transformer,
                self.model.transformer_2,
            ] if m is not None
        ]

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def _setup_ema(self, cfg: TrainConfig) -> EMA | None:
        if cfg.ema_decay <= 0:
            return None
        ema_models: dict[str, torch.nn.Module] = {}
        if cfg.train_text_encoder:
            ema_models["text_encoder"] = self.model.text_encoder
        if self.model.transformer is not None:
            ema_models["transformer"] = self.model.transformer
        if self.model.transformer_2 is not None:
            ema_models["transformer_2"] = self.model.transformer_2
        ema = EMA(ema_models, decay=cfg.ema_decay)
        logger.info("EMA enabled (decay={}, {} shadow params)", cfg.ema_decay, len(ema.shadow))
        return ema

    # ------------------------------------------------------------------
    # torch.compile
    # ------------------------------------------------------------------

    def _compile_modules(self, cfg: TrainConfig) -> None:
        compile_kwargs = {"backend": cfg.torch_compile_backend}
        if cfg.torch_compile_mode is not None:
            compile_kwargs["mode"] = cfg.torch_compile_mode
        self.model.vae = torch.compile(self.model.vae, **compile_kwargs)
        logger.info("Compiled vae")
        if not cfg.train_text_encoder:
            self.model.text_encoder = torch.compile(self.model.text_encoder, **compile_kwargs)
            logger.info("Compiled text_encoder")
        if self.model.transformer is not None:
            self.model.transformer = torch.compile(self.model.transformer, **compile_kwargs)
            logger.info("Compiled transformer")
        if self.model.transformer_2 is not None:
            self.model.transformer_2 = torch.compile(self.model.transformer_2, **compile_kwargs)
            logger.info("Compiled transformer_2")
        logger.info("torch.compile enabled (backend={}, mode={})", cfg.torch_compile_backend, cfg.torch_compile_mode)

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------

    def _build_dataset(self, cfg: TrainConfig) -> tuple:
        dataset = I2VDataset(
            json_path=cfg.dataset_json,
            num_frames=cfg.num_frames,
            max_area=cfg.max_area,
            height=cfg.height,
            width=cfg.width,
            fps=cfg.fps,
        )
        if self.expert_parallel:
            seed = self._get_expert_parallel_sampler_seed(cfg)
            sampler = DistributedSampler(
                dataset, num_replicas=self.dp_size, rank=self.dp_rank,
                shuffle=True, seed=seed,
            )
        else:
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, seed=cfg.seed,
            )
        return dataset, sampler

    def _build_dataloader(self, dataset, cfg: TrainConfig) -> StatefulDataLoader:
        kwargs = dict(
            dataset=dataset,
            batch_size=cfg.batch_size,
            sampler=self.sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate,
            drop_last=True,
        )
        if cfg.num_workers > 0:
            kwargs["persistent_workers"] = cfg.persistent_workers
            kwargs["prefetch_factor"] = cfg.prefetch_factor
        return StatefulDataLoader(**kwargs)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizers(self, cfg: TrainConfig):
        optimizer_te = None
        optimizer_1 = None
        optimizer_2 = None
        params = []
        total_params = 0
        optim_kwargs = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), fused=True)

        if cfg.train_text_encoder:
            params_te = [p for p in self.model.text_encoder.parameters() if p.requires_grad]
            params.extend(params_te)
            total_params += sum(p.numel() for p in self.model.text_encoder.parameters())
            optimizer_te = torch.optim.AdamW(params_te, **optim_kwargs)

        if self.model.transformer is not None:
            params_1 = [p for p in self.model.transformer.parameters() if p.requires_grad]
            params.extend(params_1)
            total_params += sum(p.numel() for p in self.model.transformer.parameters())
            optimizer_1 = torch.optim.AdamW(params_1, **optim_kwargs)

        if self.model.transformer_2 is not None:
            params_2 = [p for p in self.model.transformer_2.parameters() if p.requires_grad]
            params.extend(params_2)
            total_params += sum(p.numel() for p in self.model.transformer_2.parameters())
            optimizer_2 = torch.optim.AdamW(params_2, **optim_kwargs)

        trainable_count = sum(p.numel() for p in params)
        logger.info(
            "Trainable: {:.1f}M / {:.1f}M ({:.2f}%)",
            trainable_count / 1e6, total_params / 1e6, 100 * trainable_count / total_params,
        )
        optimizers = [opt for opt in [optimizer_te, optimizer_1, optimizer_2] if opt is not None]
        return params, optimizers, optimizer_te, optimizer_1, optimizer_2

    # ------------------------------------------------------------------
    # Gradient sync
    # ------------------------------------------------------------------

    def _set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        for module in self.sync_modules:
            if hasattr(module, "set_requires_gradient_sync"):
                module.set_requires_gradient_sync(requires_gradient_sync, recurse=True)

    # ------------------------------------------------------------------
    # DCP checkpointing
    # ------------------------------------------------------------------

    def _find_latest_checkpoint(self) -> str | None:
        output_dir = Path(self.cfg.output_dir)
        if not output_dir.exists():
            return None
        candidates: list[tuple[int, Path]] = []
        for d in output_dir.iterdir():
            if not d.is_dir():
                continue
            # Detect checkpoint: direct .metadata (non-EP) or EP subdirectory
            is_ckpt = (d / ".metadata").exists()
            if not is_ckpt and self.expert_parallel:
                expert_name = "high" if self.expert_group == 0 else "low"
                is_ckpt = (d / expert_name / ".metadata").exists()
            if not is_ckpt:
                continue
            name = d.name
            if name.startswith("checkpoint-epoch"):
                try:
                    int(name.removeprefix("checkpoint-epoch"))
                    candidates.append((int(d.stat().st_mtime_ns), d))
                except ValueError:
                    continue
            elif name.startswith("checkpoint-"):
                try:
                    int(name.removeprefix("checkpoint-"))
                    candidates.append((int(d.stat().st_mtime_ns), d))
                except ValueError:
                    continue
        if not candidates:
            return None
        candidates.sort()
        latest = candidates[-1][1]
        logger.info("Auto-resume: found checkpoint {}", latest)
        return str(latest)

    def _save_checkpoint(self, path: Path):
        state: dict = {"train_state": self.train_state}
        if self.ema is not None:
            state["ema"] = self.ema

        if self.expert_parallel:
            expert_name = "high" if self.expert_group == 0 else "low"
            save_path = path / expert_name
            dcp.save(state, checkpoint_id=str(save_path), process_group=self._dp_pg)
            torch.save(self.dataloader.state_dict(), save_path / f"dataloader_rank{self.dp_rank}.pt")
            if self.dp_rank == 0 and self.model.lora_config is not None:
                self.model.save_lora(str(save_path / "lora"))
            if self.dp_rank == 0:
                logger.info("Saved checkpoint to {} (expert={})", save_path, expert_name)
        else:
            dcp.save(state, checkpoint_id=str(path))
            torch.save(self.dataloader.state_dict(), path / f"dataloader_rank{self.rank}.pt")
            if self.rank == 0 and self.model.lora_config is not None:
                self.model.save_lora(str(path / "lora"))
            if self.rank == 0:
                logger.info("Saved DCP checkpoint to {}", path)
        dist.barrier()

    def _load_checkpoint(self, path: str):
        # Resolve path and DCP process group for expert parallel
        if self.expert_parallel:
            expert_name = "high" if self.expert_group == 0 else "low"
            ep_path = Path(path) / expert_name
            load_path = str(ep_path) if ep_path.exists() else path
            dl_rank = self.dp_rank
            dcp_kwargs: dict = {"process_group": self._dp_pg}
        else:
            load_path = path
            dl_rank = self.rank
            dcp_kwargs = {}

        logger.info("Resuming from {} ...", load_path)
        state: dict = {"train_state": self.train_state}
        has_legacy_ema = (Path(load_path) / "ema").is_dir()
        if self.ema is not None and not has_legacy_ema:
            state["ema"] = self.ema
        try:
            dcp.load(state, checkpoint_id=load_path, **dcp_kwargs)
        except Exception:
            if "ema" in state:
                logger.warning("Failed to load EMA from DCP, retrying without EMA")
                dcp.load({"train_state": self.train_state}, checkpoint_id=load_path, **dcp_kwargs)
            else:
                raise
        if self.ema is not None and "ema" not in state:
            self.ema.reinitialize()
            logger.warning("EMA not in DCP checkpoint, reinitialized from loaded model weights")
        if self._reset_on_load:
            self.train_state.step = 0
            self.train_state.epoch = 0
            self.train_state.batch_idx = 0
            self.params, self.optimizers, self.optimizer_te, self.optimizer_1, self.optimizer_2 = (
                self._build_optimizers(self.cfg)
            )
            self.train_state.optimizer_te = self.optimizer_te
            self.train_state.optimizer_1 = self.optimizer_1
            self.train_state.optimizer_2 = self.optimizer_2
            self.total_steps = self._compute_total_steps()
            logger.info("reset_dataloader=True: reset step/epoch/optimizer, total_steps={}", self.total_steps)
        else:
            dl_state_path = Path(load_path) / f"dataloader_rank{dl_rank}.pt"
            if dl_state_path.exists():
                self.dataloader.load_state_dict(torch.load(dl_state_path, weights_only=False))
                logger.info("Restored dataloader state from {}", dl_state_path)
        logger.info(
            "Resumed at step={} epoch={} batch_idx={}",
            self.train_state.step, self.train_state.epoch, self.train_state.batch_idx,
        )
