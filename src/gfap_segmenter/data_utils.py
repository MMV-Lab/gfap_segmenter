"""Data utilities supporting patch curation and augmentation for training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from skimage.util import view_as_windows
from tqdm import tqdm

MIN_PATCH_SIZE = 300
DEFAULT_PATCHES_PER_REGION = 1000


@dataclass
class CuratedPatch:
    image: np.ndarray
    labels: np.ndarray
    origin: Tuple[int, int]
    scale: float


def ensure_min_patch_size(
    width: int, height: int, min_size: int = MIN_PATCH_SIZE
) -> bool:
    return width >= min_size and height >= min_size


def extract_patch_from_layers(
    image: np.ndarray,
    labels: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
) -> CuratedPatch:
    y0, x0 = top_left
    y1, x1 = bottom_right
    patch_image = image[y0:y1, x0:x1]
    patch_labels = labels[y0:y1, x0:x1]
    return CuratedPatch(
        image=patch_image.copy(),
        labels=patch_labels.copy(),
        origin=(y0, x0),
        scale=1.0,
    )


def generate_augmented_patches(
    curated_patch: CuratedPatch,
    *,
    target_count: int = DEFAULT_PATCHES_PER_REGION,
    patch_size: int = MIN_PATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate augmented training samples from a curated region.

    The function returns two arrays with shapes ``(N, patch_size, patch_size)`` for
    images and masks respectively. Augmentations include random cropping,
    horizontal/vertical flips, and rotations in 90° increments.
    """

    img = curated_patch.image
    msk = curated_patch.labels

    if img.shape[0] < patch_size or img.shape[1] < patch_size:
        raise ValueError(
            "Curated patch is smaller than the required patch_size"
        )

    image_patches: List[np.ndarray] = []
    mask_patches: List[np.ndarray] = []

    rng = np.random.default_rng()
    rotations = [0, 1, 2, 3]  # multiples of 90 degrees

    # Use tqdm for progress if generating many patches
    iterator = (
        range(target_count)
        if target_count <= 100
        else tqdm(
            range(target_count),
            desc=f"Generating {target_count} patches",
            unit="patch",
            leave=False,
        )
    )

    for _ in iterator:
        y = rng.integers(0, img.shape[0] - patch_size + 1)
        x = rng.integers(0, img.shape[1] - patch_size + 1)
        crop_img = img[y : y + patch_size, x : x + patch_size]
        crop_msk = msk[y : y + patch_size, x : x + patch_size]

        if rng.random() < 0.5:
            crop_img = np.fliplr(crop_img)
            crop_msk = np.fliplr(crop_msk)
        if rng.random() < 0.5:
            crop_img = np.flipud(crop_img)
            crop_msk = np.flipud(crop_msk)

        rot_k = rng.choice(rotations)
        if rot_k:
            crop_img = np.rot90(crop_img, k=rot_k)
            crop_msk = np.rot90(crop_msk, k=rot_k)

        image_patches.append(crop_img.astype(np.float32))
        mask_patches.append(crop_msk.astype(np.uint8))

    images_array = np.stack(image_patches, axis=0)
    masks_array = np.stack(mask_patches, axis=0)
    masks_array = (masks_array > 0).astype(np.uint8)
    return images_array, masks_array


def sliding_window_view(
    image: np.ndarray, patch_size: int, stride: int
) -> np.ndarray:
    """Return a 4D view of the image using ``skimage.view_as_windows``."""

    if stride <= 0:
        raise ValueError("stride must be positive")

    windows = view_as_windows(image, (patch_size, patch_size), step=stride)
    h, w, _, _ = windows.shape
    return windows.reshape(h * w, patch_size, patch_size)


class PatchDataset(torch.utils.data.Dataset):
    """Lazy dataset backed by generated image/mask arrays."""

    def __init__(self, images: np.ndarray, masks: np.ndarray) -> None:
        self.images = torch.from_numpy(
            images[:, None, :, :].astype(np.float32)
        )
        self.masks = torch.from_numpy(masks[:, None, :, :].astype(np.float32))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.images.shape[0]

    def __getitem__(self, idx: int):  # pragma: no cover - trivial
        return self.images[idx], self.masks[idx]
