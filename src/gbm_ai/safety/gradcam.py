from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps

@dataclass
class GradCAMResult:
    cam: np.ndarray
    peak_x_normalized: float
    peak_y_normalized: float
    border_energy_fraction: float

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activation = None
        self._gradient = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        del module, inputs
        self._activation = output

    def _backward_hook(self, module, grad_input, grad_output):
        del module, grad_input
        self._gradient = grad_output[0]

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def generate(self, input_tensor: torch.Tensor) -> GradCAMResult:
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM expects input shape [1, C, H, W].")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self._activation = None
        self._gradient = None

        logits = self.model(input_tensor)
        if logits.numel() != 1:
            raise RuntimeError(f"Expected one binary logit, got {tuple(logits.shape)}.")

        logits.reshape(-1)[0].backward()

        if self._activation is None or self._gradient is None:
            raise RuntimeError("Grad-CAM hooks did not capture data.")

        weights = self._gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0].detach().float().cpu()

        lo, hi = float(cam.min()), float(cam.max())
        cam = (cam - lo) / (hi - lo) if hi - lo > 1e-12 else torch.zeros_like(cam)
        arr = cam.numpy().astype(np.float32)

        y_peak, x_peak = np.unravel_index(np.argmax(arr), arr.shape)
        h, w = arr.shape
        border = max(1, int(round(min(h, w) * 0.10)))
        mask = np.zeros((h, w), dtype=bool)
        mask[:border, :] = True
        mask[-border:, :] = True
        mask[:, :border] = True
        mask[:, -border:] = True

        total = float(arr.sum())
        border_fraction = float(arr[mask].sum() / total) if total > 1e-12 else 0.0

        return GradCAMResult(
            cam=arr,
            peak_x_normalized=float(x_peak / max(1, w - 1)),
            peak_y_normalized=float(y_peak / max(1, h - 1)),
            border_energy_fraction=border_fraction,
        )

def make_heatmap_image(cam: np.ndarray) -> Image.Image:
    rgba = colormaps["jet"](np.clip(cam, 0.0, 1.0))
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")

def make_overlay(original: Image.Image, cam: np.ndarray, alpha: float = 0.40) -> Image.Image:
    original = original.convert("RGB")
    heatmap = make_heatmap_image(cam).resize(original.size, Image.Resampling.BILINEAR)
    return Image.blend(original, heatmap, alpha=alpha)

def make_review_panel(original: Image.Image, cam: np.ndarray) -> Image.Image:
    original = original.convert("RGB")
    heatmap = make_heatmap_image(cam).resize(original.size, Image.Resampling.BILINEAR)
    overlay = make_overlay(original, cam)
    w, h = original.size
    panel = Image.new("RGB", (w * 3, h))
    panel.paste(original, (0, 0))
    panel.paste(heatmap, (w, 0))
    panel.paste(overlay, (w * 2, 0))
    return panel
