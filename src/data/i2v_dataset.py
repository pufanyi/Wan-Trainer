"""I2V training dataset.

JSON format:
[
    {
        "video": "path/to/video.mp4",             // absolute or relative to JSON dir
        "image": "ref.jpg" | ["ref.jpg"],          // optional, uses first video frame if absent
        "prompt": "description of the video"
    }
]
"""

import json
from pathlib import Path

import decord
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

decord.bridge.set_bridge("torch")


class I2VDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        fps: int = 16,
    ):
        json_path = Path(json_path)
        self.base_dir = json_path.parent
        self.data = json.loads(json_path.read_text())
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps

    def __len__(self):
        return len(self.data)

    def _resolve(self, path: str) -> str:
        """Resolve path: absolute stays absolute, relative resolves from JSON dir."""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(self.base_dir / p)

    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load video frames as uint8. Returns (C, T, H, W)."""
        vr = decord.VideoReader(video_path, width=self.width, height=self.height)
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)

        # Sample frames at target fps
        stride = max(1, round(video_fps / self.fps))
        indices = list(range(0, total_frames, stride))[: self.num_frames]

        # Pad if not enough frames by repeating last
        while len(indices) < self.num_frames:
            indices.append(indices[-1])

        frames = vr.get_batch(indices)  # (T, H, W, C) uint8 torch tensor
        return frames.permute(3, 0, 1, 2).contiguous()

    def _load_image(self, path: str) -> torch.Tensor:
        """Load a single image as uint8. Returns (C, H, W)."""
        with Image.open(path) as img:
            img = img.convert("RGB").resize((self.width, self.height), Image.LANCZOS)
            array = np.asarray(img, dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx):
        item = self.data[idx]
        video = self._load_video(self._resolve(item["video"]))  # (C, T, H, W)

        # Reference image: string, list (use first element), or absent (first video frame)
        raw_image = item.get("image")
        if isinstance(raw_image, list):
            raw_image = raw_image[0] if raw_image else None
        image = self._load_image(self._resolve(raw_image)) if raw_image else video[:, 0].clone()

        return {
            "index": idx,
            "video": video,
            "image": image,
            "prompt": item["prompt"],
        }
