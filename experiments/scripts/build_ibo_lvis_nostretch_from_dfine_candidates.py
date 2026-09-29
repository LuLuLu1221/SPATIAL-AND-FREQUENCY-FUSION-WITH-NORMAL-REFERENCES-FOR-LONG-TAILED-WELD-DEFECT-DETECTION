#!/usr/bin/env python
"""LVIS/SimLTD no-stretch IBO candidate builder.

This is a thin adapter around the existing IBO candidate script.  The original
script was written for weld-strip inputs and used a direct square resize during
D-FINE inference.  For the SimLTD/LVIS experiment we must keep the no-stretch
setting used during training: resize by the longest side, pad to 640, run
D-FINE, then project candidate boxes back to the original image coordinates.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF


LEGACY_IBO_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "方案代码_150px_DFINE_IBO_NRef_Freq_Confusion_20260915"
    / "scripts"
    / "build_ibo_from_dfine_candidates.py"
)


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_ibo_builder", LEGACY_IBO_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy IBO script: {LEGACY_IBO_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model_no_stretch(spec, dfine_root: Path, scripts_root: Path, device: torch.device):
    """Build D-FINE and return original-coordinate boxes after ResizeMaxPad."""
    sys.path.insert(0, str(dfine_root))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(spec.config), resume=str(spec.checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    spatial_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    if not isinstance(spatial_size, (list, tuple)) or len(spatial_size) != 2:
        raise ValueError(f"{spec.name}: eval_spatial_size must be [height, width], got {spatial_size!r}")
    eval_h, eval_w = [int(value) for value in spatial_size]
    if eval_h != eval_w:
        raise ValueError("This LVIS no-stretch adapter currently expects square eval_spatial_size.")
    eval_size = eval_h

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images: torch.Tensor, original_sizes: torch.Tensor):
            raw_outputs = self.model(images)
            padded_sizes = torch.full(
                (images.shape[0], 2),
                float(eval_size),
                dtype=torch.float32,
                device=images.device,
            )
            labels, boxes, scores = self.postprocessor(raw_outputs, padded_sizes)

            # Convert from padded 640x640 coordinates to original image coords.
            widths = original_sizes[:, 0].to(boxes.device)
            heights = original_sizes[:, 1].to(boxes.device)
            max_sides = torch.maximum(widths, heights)
            scales = float(eval_size) / max_sides
            valid_widths = widths * scales
            valid_heights = heights * scales

            boxes = boxes.clone()
            boxes[..., 0::2] = boxes[..., 0::2].clamp(min=0.0)
            boxes[..., 1::2] = boxes[..., 1::2].clamp(min=0.0)
            boxes[..., 0::2] = torch.minimum(boxes[..., 0::2], valid_widths[:, None, None])
            boxes[..., 1::2] = torch.minimum(boxes[..., 1::2], valid_heights[:, None, None])
            boxes = boxes / scales[:, None, None]
            boxes[..., 0::2] = torch.minimum(boxes[..., 0::2], widths[:, None, None])
            boxes[..., 1::2] = torch.minimum(boxes[..., 1::2], heights[:, None, None])
            return labels, boxes, scores

    def transform(image):
        width, height = image.size
        scale = eval_size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = TVF.resize(image, [new_height, new_width])
        tensor = TVF.to_tensor(resized)
        pad_right = eval_size - new_width
        pad_bottom = eval_size - new_height
        return F.pad(tensor, [0, pad_right, 0, pad_bottom], value=0.0)

    return Model().to(device).eval(), transform, [eval_h, eval_w]


def main() -> None:
    legacy = load_legacy_module()
    legacy.build_model = build_model_no_stretch
    legacy.main()


if __name__ == "__main__":
    main()
