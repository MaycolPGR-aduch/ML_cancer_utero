"""
Grad-CAM nativo en PyTorch para explicabilidad (XAI) del modulo CNN MobileNetV2.
Permite superponer un mapa de calor sobre la imagen citologica de 224x224
mostrando las regiones celulares que mas influyeron en la prediccion.
"""

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class MobileNetV2GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.target_layer = self.model.features[-1]
        self.activations = None
        self.gradients = None
        self._handles = []

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inp, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_pil: Image.Image,
        target_class: int | None = None,
        alpha: float = 0.45,
    ) -> Tuple[np.ndarray, int]:
        """
        Retorna (overlay_rgb_uint8_224x224, target_class_idx).
        """
        self.model.eval()
        self._register_hooks()
        try:
            self.model.zero_grad(set_to_none=True)
            inp = input_tensor.clone().detach().requires_grad_(True)
            logits = self.model(inp)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())
            score = logits[0, target_class]
            score.backward()

            # Pesos globales por canal (Global Average Pooling de gradientes)
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam).squeeze().cpu().numpy()

            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros_like(cam)

            cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
            heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            base_rgb = np.array(original_pil.convert("RGB").resize((224, 224), Image.BILINEAR))
            overlay = np.uint8(np.clip((1.0 - alpha) * base_rgb + alpha * heatmap_rgb, 0, 255))
            return overlay, target_class
        finally:
            self._remove_hooks()
