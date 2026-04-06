"""I2V training dataset.

JSON format (list of items — legacy):
[
    {
        "video": "path/to/video.mp4",             // absolute or relative to JSON dir
        "image": "ref.jpg" | ["ref.jpg"],          // optional, uses first video frame if absent
        "prompt": "description of the video",
        "search_video": "path/to/search.mp4"      // optional, for COS training
    }
]

JSON format (dict — single dataset config):
{
    "num_frames": 81,
    "max_area": 184320,       // height * width budget
    "height": 320,            // optional fixed height (overrides max_area)
    "width": 576,             // optional fixed width (overrides max_area)
    "fps": 16,
    "root": "/abs/path/to/data/dir",  // optional, base dir for relative paths in items
    "data": [ ... ]           // same item format as above
    // OR
    "data_path": "path/to/items.json"  // load items from external file
}

JSON format (list of dataset configs — multi-dataset):
[
    {
        "num_frames": 81,
        "max_area": 184320,
        "fps": 16,
        "root": "/abs/path/to/data/dir",    // optional, base dir for relative paths in items
        "data_path": "path/to/items.json"   // or "data": [...]
    },
    ...
]
"""

import json
import logging
import random
from pathlib import Path

import decord
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

decord.bridge.set_bridge("torch")

# Height/width must be divisible by vae_scale_factor_spatial * patch_size.
# For Wan2.2: 8 * 2 = 16.
_MOD_VALUE = 16

# Config keys that distinguish a dataset config dict from a data item dict.
_CONFIG_KEYS = {"num_frames", "max_area", "fps", "data", "data_path", "height", "width", "root"}


class _ItemConfig(BaseModel):
    num_frames: int
    max_area: int
    fixed_height: int | None = None
    fixed_width: int | None = None
    fps: int


def compute_hw(max_area: int, aspect_ratio: float) -> tuple[int, int]:
    """Compute (height, width) from a pixel budget and aspect ratio (h/w)."""
    height = round(np.sqrt(max_area * aspect_ratio)) // _MOD_VALUE * _MOD_VALUE
    width = round(np.sqrt(max_area / aspect_ratio)) // _MOD_VALUE * _MOD_VALUE
    # Ensure at least _MOD_VALUE
    height = max(height, _MOD_VALUE)
    width = max(width, _MOD_VALUE)
    return height, width


def _is_config_dict(d: dict) -> bool:
    """Check if a dict looks like a dataset config (vs a data item)."""
    return bool(set(d.keys()) & _CONFIG_KEYS)


class I2VDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        num_frames: int | None = None,
        max_area: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
    ):
        json_path_converted = Path(json_path)
        base_dir = json_path_converted.parent
        raw = json.loads(json_path_converted.read_text())

        # Collect (item, base_dir, config_dict) tuples then build final lists.
        items: list[dict] = []
        item_configs: list[_ItemConfig] = []
        base_dirs: list[Path] = []

        if isinstance(raw, dict):
            # Single dataset config dict
            cfg_items, cfg_base = self._load_config_items(raw, base_dir)
            cfg = self._make_item_config(raw, num_frames, max_area, height, width, fps)
            items.extend(cfg_items)
            item_configs.extend([cfg] * len(cfg_items))
            base_dirs.extend([cfg_base] * len(cfg_items))
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict) and _is_config_dict(raw[0]):
            # List of dataset configs (multi-dataset)
            for entry in raw:
                cfg_items, cfg_base = self._load_config_items(entry, base_dir)
                cfg = self._make_item_config(entry, num_frames, max_area, height, width, fps)
                items.extend(cfg_items)
                item_configs.extend([cfg] * len(cfg_items))
                base_dirs.extend([cfg_base] * len(cfg_items))
        else:
            # Legacy: plain list of data items
            default_cfg = self._make_item_config({}, num_frames, max_area, height, width, fps)
            items.extend(raw)
            item_configs.extend([default_cfg] * len(raw))
            base_dirs.extend([base_dir] * len(raw))

        self.data = items
        self._item_configs = item_configs
        self._base_dirs = base_dirs

    @staticmethod
    def _load_config_items(cfg: dict, parent_dir: Path) -> tuple[list[dict], Path]:
        """Load data items from a config dict. Returns (items, base_dir).

        base_dir priority: explicit "root" > data_path parent > parent_dir.
        """
        if "data" in cfg:
            items = cfg["data"]
            default_base = parent_dir
        elif "data_path" in cfg:
            data_path = Path(cfg["data_path"])
            if not data_path.is_absolute():
                data_path = parent_dir / data_path
            items = json.loads(data_path.read_text())
            default_base = data_path.parent
        else:
            raise ValueError(f"Dataset config must contain 'data' or 'data_path': {cfg}")

        # Explicit root overrides the default base directory.
        if "root" in cfg:
            root = Path(cfg["root"])
            if not root.is_absolute():
                root = parent_dir / root
            return items, root
        return items, default_base

    @staticmethod
    def _make_item_config(
        json_cfg: dict,
        num_frames: int | None,
        max_area: int | None,
        height: int | None,
        width: int | None,
        fps: int | None,
    ) -> _ItemConfig:
        """Build per-item config. Priority: constructor param > JSON config > default."""
        return _ItemConfig(
            num_frames=num_frames if num_frames is not None else json_cfg.get("num_frames", 81),
            max_area=max_area if max_area is not None else json_cfg.get("max_area", 480 * 832),
            fixed_height=height if height is not None else json_cfg.get("height"),
            fixed_width=width if width is not None else json_cfg.get("width"),
            fps=fps if fps is not None else json_cfg.get("fps", 16),
        )

    def __len__(self):
        return len(self.data)

    def _resolve(self, path: str, base_dir: Path) -> str:
        """Resolve path: absolute stays absolute, relative resolves from base_dir."""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(base_dir / p)

    @staticmethod
    def _get_video_hw(video_path: str, cfg: _ItemConfig) -> tuple[int, int]:
        """Return target (height, width). Uses fixed h/w if set, otherwise derives from video aspect ratio."""
        if cfg.fixed_height is not None and cfg.fixed_width is not None:
            return cfg.fixed_height, cfg.fixed_width
        vr = decord.VideoReader(video_path)
        orig_h, orig_w = vr[0].shape[:2]
        return compute_hw(cfg.max_area, orig_h / orig_w)

    @staticmethod
    def _load_video(video_path: str, height: int, width: int, cfg: _ItemConfig) -> torch.Tensor:
        """Load video frames as uint8. Returns (C, T, H, W)."""
        vr = decord.VideoReader(video_path, width=width, height=height)
        total_frames = len(vr)

        # Uniformly sample num_frames across the entire video (stretch or compress)
        indices = np.linspace(0, total_frames - 1, cfg.num_frames).round().astype(int).tolist()

        frames = vr.get_batch(indices)  # (T, H, W, C) uint8 torch tensor
        return frames.permute(3, 0, 1, 2).contiguous()

    @staticmethod
    def _load_image(path: str, height: int, width: int) -> torch.Tensor:
        """Load a single image as uint8. Returns (C, H, W)."""
        with Image.open(path) as img:
            img = img.convert("RGB").resize((width, height), Image.LANCZOS)
            array = np.array(img, dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    _MAX_RETRIES = 10

    def __getitem__(self, idx):
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._load_item(idx)
            except Exception:
                logger.warning(
                    "Failed to load item %d (attempt %d/%d), trying another sample.",
                    idx,
                    attempt + 1,
                    self._MAX_RETRIES,
                    exc_info=True,
                )
                idx = random.randint(0, len(self.data) - 1)
        # Final attempt — let it raise if it still fails.
        return self._load_item(idx)

    def _load_item(self, idx):
        item = self.data[idx]
        cfg = self._item_configs[idx]
        base_dir = self._base_dirs[idx]

        video_path = self._resolve(item["video"], base_dir)
        height, width = self._get_video_hw(video_path, cfg)
        video = self._load_video(video_path, height, width, cfg)  # (C, T, H, W)

        # Reference image: string, list (use first element), or absent (first video frame)
        raw_image = item.get("image")
        if isinstance(raw_image, list):
            raw_image = raw_image[0] if raw_image else None
        image = (
            self._load_image(self._resolve(raw_image, base_dir), height, width) if raw_image else video[:, 0].clone()
        )

        result = {
            "index": idx,
            "video": video,
            "image": image,
            "prompt": item["prompt"],
        }

        # Optional search_video for COS training
        if "search_video" in item:
            search_path = self._resolve(item["search_video"], base_dir)
            result["search_video"] = self._load_video(search_path, height, width, cfg)

        return result
