"""I2V training dataset.

JSON format:
[
    {
        "video": "path/to/video.mp4",
        "image": "path/to/first_frame.jpg",  // optional, uses first video frame if absent
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
        self.data = json.loads(Path(json_path).read_text())
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps

    def __len__(self):
        return len(self.data)

    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load video frames, resized and normalized to [-1, 1]. Returns (C, T, H, W)."""
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
        video = frames.permute(3, 0, 1, 2).float() / 127.5 - 1.0  # (C, T, H, W) in [-1, 1]
        return video

    def _load_image(self, path: str) -> torch.Tensor:
        """Load a single image, resized and normalized to [-1, 1]. Returns (C, H, W)."""
        img = Image.open(path).convert("RGB").resize((self.width, self.height), Image.LANCZOS)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0

    def __getitem__(self, idx):
        item = self.data[idx]
        video = self._load_video(item["video"])  # (C, T, H, W)

        # First frame: use explicit image if provided, else extract from video
        if "image" in item and item["image"]:
            image = self._load_image(item["image"])
        else:
            image = video[:, 0].clone()  # (C, H, W)

        return {
            "video": video,
            "image": image,
            "prompt": item["prompt"],
        }
