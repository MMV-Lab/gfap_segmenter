"""Model management utilities for the GFAP napari plugin.

The :class:`ModelManager` centralises access to the default GFAP U-Net
checkpoint, provides helpers for importing custom checkpoints/ONNX models, and
exposes a high-level ``predict`` API that performs tiled inference on large
2D images. The goal is to keep all heavy-lifting outside of the Qt event loop
so that the widget can trigger work without blocking the UI.

Notes
-----
* The default checkpoint path is resolved relative to the repository; when the
  file is not present, ``load_default_model`` raises a descriptive error.
* When ONNX Runtime is available we support loading and exporting ONNX models
  in addition to native PyTorch checkpoints.
* The tiled inference strategy mirrors the standalone ``predict.py`` script,
  using a Hann window to blend overlapping predictions and minimise edge
  artefacts.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:  # pragma: no cover - optional dependency
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except Exception:  # pragma: no cover
    onnx = None  # type: ignore[assignment]
    ort = None  # type: ignore[assignment]
    ONNX_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from onnxruntime import InferenceSession
else:  # pragma: no cover - fallback type
    InferenceSession = object


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "default_model.ckpt"
)


def _get_default_device() -> torch.device:
    if torch.cuda.is_available():  # pragma: no cover - GPU specific
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - macOS
        return torch.device("mps")
    return torch.device("cpu")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Match original implementation: ConvTranspose2d takes in_channels -> in_channels // 2
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            # After ConvTranspose2d and concat, input to DoubleConv is in_channels channels
            # Use default mid_channels=out_channels to match checkpoint structure
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Tuple[int, int, int, int] | None = None,
        num_fmaps: int | None = None,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        # Support both features tuple and num_fmaps (for compatibility)
        if features is None:
            if num_fmaps is None:
                num_fmaps = 64  # Default
            features = (num_fmaps, num_fmaps * 2, num_fmaps * 4, num_fmaps * 8)

        # Match original UNet structure exactly
        num_fmaps = features[0]  # First feature map size
        self.inc = DoubleConv(in_channels, num_fmaps)
        self.down1 = Down(num_fmaps, num_fmaps * 2)
        self.down2 = Down(num_fmaps * 2, num_fmaps * 4)
        self.down3 = Down(num_fmaps * 4, num_fmaps * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(num_fmaps * 8, num_fmaps * 16 // factor)
        self.up1 = Up(num_fmaps * 16, num_fmaps * 8 // factor, bilinear)
        self.up2 = Up(num_fmaps * 8, num_fmaps * 4 // factor, bilinear)
        self.up3 = Up(num_fmaps * 4, num_fmaps * 2 // factor, bilinear)
        self.up4 = Up(num_fmaps * 2, num_fmaps, bilinear)
        self.outc = OutConv(num_fmaps, out_channels)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - tested via manager
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


@dataclass
class PredictionResult:
    mask: np.ndarray
    probability: np.ndarray


class ModelManager:
    """Handle model loading, predictions, and export operations."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or _get_default_device()
        self.model_path = Path(model_path) if model_path else None
        self.model: nn.Module | None = None
        self.onnx_session: InferenceSession | None = None
        self.model_metadata: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Device information
    # ------------------------------------------------------------------
    def get_device_info(self) -> Dict[str, str]:
        """Return device information for display in the UI.

        Returns:
            Dictionary with keys:
            - 'type': Device type string (e.g., 'cuda', 'cpu', 'mps')
            - 'name': Human-readable device name (e.g., 'NVIDIA GeForce RTX 3090')
            - 'available': Whether the device is actually available
        """
        device_type = str(self.device)
        device_name = "Unknown"
        available = True

        if self.device.type == "cuda":
            if torch.cuda.is_available():
                try:
                    device_name = torch.cuda.get_device_name(0)
                except Exception:
                    device_name = "CUDA GPU"
                available = True
            else:
                device_name = "CUDA unavailable"
                available = False
        elif self.device.type == "mps":
            if torch.backends.mps.is_available():
                device_name = "Apple Silicon GPU"
                available = True
            else:
                device_name = "MPS unavailable"
                available = False
        elif self.device.type == "cpu":
            device_name = "CPU"
            available = True

        return {
            "type": device_type,
            "name": device_name,
            "available": str(available),
        }

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load_default_model(self) -> None:
        if not DEFAULT_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Default checkpoint not found. Expected at "
                f"{DEFAULT_MODEL_PATH}."
            )
        self.load_checkpoint(DEFAULT_MODEL_PATH)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            if any(key.startswith("model.") for key in state_dict):
                state_dict = {
                    key.replace("model.", "", 1): value
                    for key, value in state_dict.items()
                }
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:  # pragma: no cover - unusual checkpoint format
            raise ValueError("Unsupported checkpoint format")

        model_config = self._extract_model_config(checkpoint)

        # Double-check bilinear detection from state_dict directly
        # This is critical - check the actual state_dict we'll use
        # Keys are like "up1.up.weight", "up2.up.weight", etc. (ConvTranspose2d when bilinear=False)
        # When bilinear=True, there's no "up.weight" key (uses Upsample instead)
        has_conv_transpose = any(
            ".up.weight" in k or ".up.bias" in k for k in state_dict.keys()
        )
        if has_conv_transpose:
            model_config["bilinear"] = False
        elif model_config.get("bilinear") is None:
            model_config["bilinear"] = True  # Default fallback

        # Ensure bilinear is a boolean
        model_config["bilinear"] = bool(model_config.get("bilinear", True))

        # Filter out None values before passing to UNet, but keep bilinear explicitly
        filtered_config = {
            k: v for k, v in model_config.items() if v is not None
        }
        # Ensure bilinear is explicitly set (don't rely on default)
        filtered_config["bilinear"] = model_config["bilinear"]
        # Debug: log what we're creating
        import logging

        logging.debug(f"Creating UNet with config: {filtered_config}")
        self.model = UNet(**filtered_config)

        # Remap state_dict keys for compatibility with different checkpoint formats
        bilinear = model_config["bilinear"]
        state_dict = self._remap_state_dict(state_dict, bilinear)

        # Use strict=False to handle minor key mismatches, but log warnings
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys:
            import warnings

            warnings.warn(
                f"Missing keys when loading checkpoint: {missing_keys[:5]}..."
                if len(missing_keys) > 5
                else f"Missing keys: {missing_keys}"
            )
        if unexpected_keys:
            import warnings

            warnings.warn(
                f"Unexpected keys in checkpoint (ignored): {unexpected_keys[:5]}..."
                if len(unexpected_keys) > 5
                else f"Unexpected keys: {unexpected_keys}"
            )
        self.model.to(self.device)
        self.model.eval()
        self.model_path = checkpoint_path
        self.onnx_session = None
        self.model_metadata = {
            "source": str(checkpoint_path),
            "backend": "torch",
        }

    def load_onnx(self, path: Path) -> None:
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required to load ONNX models")

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        providers = ort.get_available_providers()
        self.onnx_session = ort.InferenceSession(
            str(model_path), providers=providers
        )
        self.model = None
        self.model_path = model_path
        self.model_metadata = {
            "source": str(model_path),
            "backend": "onnx",
            "providers": providers,
        }

    def load_model(self, path: Path) -> None:
        suffix = Path(path).suffix.lower()
        if suffix in {".ckpt", ".pt", ".pth"}:
            self.load_checkpoint(path)
        elif suffix == ".onnx":
            self.load_onnx(path)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported model format: {suffix}")

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict(
        self,
        image: np.ndarray,
        *,
        patch_size: int = 256,
        overlap: int = 32,
        batch_size: int = 4,
        threshold: float = 0.5,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PredictionResult:
        if self.model is None and self.onnx_session is None:
            raise RuntimeError(
                "No model loaded. Call load_default_model or load_model first."
            )

        start_time = time.time()
        print(f"[PREDICT] Starting prediction on image shape: {image.shape}")

        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(
                "ModelManager.predict expects a 2D single-channel image"
            )

        # Normalize to [0, 1] (match predict.py behavior)
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        # If image_max == image_min, image is already constant, no normalization needed

        # Phase 1: Tiling (0-10% of progress)
        if progress_callback:
            progress_callback(0.0)
        t0 = time.time()
        tiles, stitcher = _tile_image(
            image, patch_size=patch_size, overlap=overlap
        )
        tiling_time = time.time() - t0
        print(
            f"[PREDICT] Tiling completed in {tiling_time:.2f}s - Created {len(tiles)} tiles"
        )
        if progress_callback:
            progress_callback(0.1)  # 10% after tiling

        # Phase 2: Inference (10-90% of progress)
        t0 = time.time()

        # Create a wrapper callback that maps inference progress (0-1) to overall progress (0.1-0.9)
        def inference_progress(value: float) -> None:
            if progress_callback:
                progress_callback(0.1 + 0.8 * value)  # Map [0,1] to [0.1, 0.9]

        probabilities = self._predict_tiles(
            tiles,
            batch_size=batch_size,
            progress_callback=inference_progress,
        )
        prediction_time = time.time() - t0
        print(f"[PREDICT] Model inference completed in {prediction_time:.2f}s")

        # Phase 3: Stitching (90-95% of progress)
        if progress_callback:
            progress_callback(0.9)
        t0 = time.time()
        probability_map = stitcher(probabilities)
        stitching_time = time.time() - t0
        print(f"[PREDICT] Stitching completed in {stitching_time:.2f}s")
        if progress_callback:
            progress_callback(0.95)

        # Phase 4: Thresholding (95-100% of progress)
        t0 = time.time()
        mask = (probability_map >= threshold).astype(np.uint8)
        threshold_time = time.time() - t0

        total_time = time.time() - start_time
        print(f"[PREDICT] Thresholding completed in {threshold_time:.2f}s")
        print(
            f"[PREDICT] Total prediction time: {total_time:.2f}s (tiling: {tiling_time:.2f}s, inference: {prediction_time:.2f}s, stitching: {stitching_time:.2f}s, threshold: {threshold_time:.2f}s)"
        )

        if progress_callback:
            progress_callback(1.0)  # 100% complete

        return PredictionResult(mask=mask, probability=probability_map)

    def _predict_tiles(
        self,
        tiles: np.ndarray,
        *,
        batch_size: int,
        progress_callback: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        if self.onnx_session is not None:
            return self._predict_tiles_onnx(
                tiles,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )
        assert self.model is not None
        start_time = time.time()
        outputs: list[np.ndarray] = []
        total_batches = int(np.ceil(len(tiles) / batch_size))
        print(
            f"[PREDICT_TILES] Starting PyTorch inference: {len(tiles)} tiles, batch_size={batch_size}, total_batches={total_batches}, device={self.device}"
        )

        batch_times = []
        with torch.no_grad():
            for batch_index, batch in enumerate(
                _iter_batches(tiles, batch_size)
            ):
                batch_start = time.time()
                tensor = torch.from_numpy(batch).to(self.device)
                preds = torch.sigmoid(self.model(tensor))
                outputs.append(preds.cpu().numpy())
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                if progress_callback:
                    progress_callback((batch_index + 1) / total_batches)
                if (batch_index + 1) % max(1, total_batches // 10) == 0 or (
                    batch_index + 1
                ) == total_batches:
                    avg_batch_time = sum(batch_times) / len(batch_times)
                    elapsed = time.time() - start_time
                    print(
                        f"[PREDICT_TILES] Batch {batch_index + 1}/{total_batches} completed - batch time: {batch_time:.3f}s, avg: {avg_batch_time:.3f}s, elapsed: {elapsed:.1f}s"
                    )

        t0 = time.time()
        result = np.concatenate(outputs, axis=0)
        concat_time = time.time() - t0
        total_time = time.time() - start_time
        print(f"[PREDICT_TILES] Concatenation completed in {concat_time:.3f}s")
        print(
            f"[PREDICT_TILES] Total inference time: {total_time:.2f}s, avg batch time: {sum(batch_times)/len(batch_times):.3f}s"
        )
        return result

    def _predict_tiles_onnx(
        self,
        tiles: np.ndarray,
        *,
        batch_size: int,
        progress_callback: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        assert self.onnx_session is not None
        start_time = time.time()
        outputs: list[np.ndarray] = []
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        total_batches = int(np.ceil(len(tiles) / batch_size))
        print(
            f"[PREDICT_TILES_ONNX] Starting ONNX inference: {len(tiles)} tiles, batch_size={batch_size}, total_batches={total_batches}"
        )

        batch_times = []
        for batch_index, batch in enumerate(_iter_batches(tiles, batch_size)):
            batch_start = time.time()
            preds = self.onnx_session.run([output_name], {input_name: batch})[
                0
            ]
            outputs.append(preds)
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            if progress_callback:
                progress_callback((batch_index + 1) / total_batches)
            if (batch_index + 1) % max(1, total_batches // 10) == 0 or (
                batch_index + 1
            ) == total_batches:
                avg_batch_time = sum(batch_times) / len(batch_times)
                elapsed = time.time() - start_time
                print(
                    f"[PREDICT_TILES_ONNX] Batch {batch_index + 1}/{total_batches} completed - batch time: {batch_time:.3f}s, avg: {avg_batch_time:.3f}s, elapsed: {elapsed:.1f}s"
                )

        t0 = time.time()
        result = np.concatenate(outputs, axis=0)
        concat_time = time.time() - t0
        total_time = time.time() - start_time
        print(
            f"[PREDICT_TILES_ONNX] Concatenation completed in {concat_time:.3f}s"
        )
        print(
            f"[PREDICT_TILES_ONNX] Total inference time: {total_time:.2f}s, avg batch time: {sum(batch_times)/len(batch_times):.3f}s"
        )
        return result

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def export_onnx(
        self,
        output_path: Path,
        input_shape: Tuple[int, int, int, int] = (1, 1, 512, 512),
    ) -> Path:
        if self.model is None:
            raise RuntimeError(
                "A PyTorch model must be loaded before exporting ONNX"
            )
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnx and onnxruntime are required for ONNX export"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(*input_shape, device=self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
        )
        return output_path

    # ------------------------------------------------------------------
    def _remap_state_dict(
        self, state_dict: Dict[str, torch.Tensor], bilinear: bool
    ) -> Dict[str, torch.Tensor]:
        """Remap state_dict keys to match current model structure.

        Handles:
        - outc.weight/bias -> outc.conv.weight/bias
        - Removes up.up.weight/bias when bilinear=True (Upsample has no weights)
        """
        remapped = {}
        for key, value in state_dict.items():
            # Handle OutConv: outc.weight -> outc.conv.weight
            if key == "outc.weight":
                remapped["outc.conv.weight"] = value
            elif key == "outc.bias":
                remapped["outc.conv.bias"] = value
            # Skip up.up weights when bilinear=True (Upsample doesn't have weights)
            elif bilinear and ("up.up.weight" in key or "up.up.bias" in key):
                continue  # Skip ConvTranspose2d weights when using Upsample
            else:
                remapped[key] = value
        return remapped

    def _extract_model_config(
        self, checkpoint: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        config = {
            "in_channels": 1,
            "out_channels": 1,
            "bilinear": None,  # None means we need to infer it
            "num_fmaps": None,
        }

        if isinstance(checkpoint, dict):
            hyper_params = checkpoint.get("hyper_parameters")
            if isinstance(hyper_params, dict):
                cfg = hyper_params.get("config", {}).get("model", {})
                # Update with any matching keys
                for key in [
                    "in_channels",
                    "out_channels",
                    "bilinear",
                    "num_fmaps",
                ]:
                    if key in cfg:
                        config[key] = cfg[key]
                # Also check for num_fmaps at top level of model config
                if "num_fmaps" not in config or config["num_fmaps"] is None:
                    if "num_fmaps" in cfg:
                        config["num_fmaps"] = cfg["num_fmaps"]

        # Infer config from state_dict if not available
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Infer bilinear from state_dict (check this first, before model creation)
        if config.get("bilinear") is None:
            # If we see .up.weight keys (like "up1.up.weight"), it means bilinear=False (ConvTranspose2d)
            # When bilinear=True, there's no "up.weight" key (uses Upsample instead)
            has_conv_transpose = any(
                ".up.weight" in k or ".up.bias" in k for k in state_dict.keys()
            )
            config["bilinear"] = not has_conv_transpose

        # Ensure bilinear is a boolean, not None
        if config.get("bilinear") is None:
            config["bilinear"] = True  # Safe default

        # Infer num_fmaps from state_dict if not in config
        if config.get("num_fmaps") is None:
            # Check the first conv layer size: inc.double_conv.0.weight shape is [num_fmaps, in_channels, 3, 3]
            inc_key = "inc.double_conv.0.weight"
            if inc_key in state_dict:
                num_fmaps = state_dict[inc_key].shape[0]
                config["num_fmaps"] = num_fmaps
            else:
                # Fallback: try to infer from other layers
                for key in state_dict.keys():
                    if (
                        "inc" in key
                        and "weight" in key
                        and len(state_dict[key].shape) == 4
                    ):
                        config["num_fmaps"] = state_dict[key].shape[0]
                        break
                # Ultimate fallback: default to 32 (common for GFAP models)
                if config.get("num_fmaps") is None:
                    config["num_fmaps"] = 32

        return config


def _iter_batches(tiles: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start : start + batch_size]
        yield batch


def _tile_image(
    image: np.ndarray,
    *,
    patch_size: int,
    overlap: int,
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Split the image into overlapping tiles and return a stitcher callback."""
    start_time = time.time()
    print(
        f"[TILING] Starting tiling: image shape={image.shape}, patch_size={patch_size}, overlap={overlap}"
    )

    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than patch_size")

    t0 = time.time()
    padded_height = (
        int(np.ceil((image.shape[0] - overlap) / stride)) * stride + overlap
    )
    padded_width = (
        int(np.ceil((image.shape[1] - overlap) / stride)) * stride + overlap
    )

    pad_height = max(0, padded_height - image.shape[0])
    pad_width = max(0, padded_width - image.shape[1])
    padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode="reflect")
    pad_time = time.time() - t0
    print(
        f"[TILING] Padding completed in {pad_time:.3f}s - padded shape: {padded.shape}"
    )

    t0 = time.time()
    tiles = []
    positions = []
    num_tiles_y = len(range(0, padded.shape[0] - overlap, stride))
    num_tiles_x = len(range(0, padded.shape[1] - overlap, stride))
    for y in range(0, padded.shape[0] - overlap, stride):
        for x in range(0, padded.shape[1] - overlap, stride):
            tile = padded[y : y + patch_size, x : x + patch_size]
            tiles.append(tile)
            positions.append((y, x))
    tile_extraction_time = time.time() - t0
    print(
        f"[TILING] Tile extraction completed in {tile_extraction_time:.3f}s - {len(tiles)} tiles ({num_tiles_y}x{num_tiles_x})"
    )

    t0 = time.time()
    tiles_array = np.array(tiles, dtype=np.float32)
    tiles_array = tiles_array[:, None, :, :]
    array_creation_time = time.time() - t0
    print(
        f"[TILING] Array creation completed in {array_creation_time:.3f}s - tiles_array shape: {tiles_array.shape}"
    )

    window = _hann_window(patch_size)
    total_tiling_time = time.time() - start_time
    print(f"[TILING] Tiling total time: {total_tiling_time:.3f}s")

    def stitch(probabilities: np.ndarray) -> np.ndarray:
        stitch_start = time.time()
        print(
            f"[STITCHING] Starting stitching: {len(probabilities)} probability tiles"
        )
        canvas = np.zeros_like(padded, dtype=np.float32)
        weight = np.zeros_like(padded, dtype=np.float32)

        t0 = time.time()
        for (y, x), prob in zip(positions, probabilities):
            prob_2d = prob.squeeze()
            canvas[y : y + patch_size, x : x + patch_size] += prob_2d * window
            weight[y : y + patch_size, x : x + patch_size] += window
        blending_time = time.time() - t0
        print(f"[STITCHING] Tile blending completed in {blending_time:.3f}s")

        t0 = time.time()
        weight[weight == 0] = 1.0
        stitched = canvas / weight
        normalization_time = time.time() - t0
        print(
            f"[STITCHING] Normalization completed in {normalization_time:.3f}s"
        )

        result = stitched[: image.shape[0], : image.shape[1]]
        total_stitch_time = time.time() - stitch_start
        print(f"[STITCHING] Stitching total time: {total_stitch_time:.3f}s")
        return result

    return tiles_array, stitch


def _hann_window(size: int) -> np.ndarray:
    hann_1d = np.hanning(size)
    window = np.outer(hann_1d, hann_1d)
    window = window / window.max()
    return window.astype(np.float32)
