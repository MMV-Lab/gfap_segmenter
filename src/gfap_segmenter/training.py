"""Model training utilities for the GFAP napari plugin."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data_utils import CuratedPatch, PatchDataset, generate_augmented_patches
from .model_manager import ModelManager, UNet


class TrainingManager:
    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager
        self.device = model_manager.device
        self.models_dir = Path.home() / ".gfap_segmenter" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_from_curated_patches(
        self,
        patches: List[CuratedPatch],
        *,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        progress_callback: Callable[[float], None] | None = None,
        message_callback: Callable[[str], None] | None = None,
    ) -> str:
        # Phase 1: Setup (0-30% of progress)
        if progress_callback:
            progress_callback(0.0)

        print("[TRAINING] Starting training preparation...")
        print(f"[TRAINING] Curated patches: {len(patches)}")

        # Phase 1a: Patch generation (0-20% of progress)
        def patch_progress(value: float) -> None:
            if progress_callback:
                progress_callback(0.2 * value)  # Map [0,1] to [0, 0.2]

        images, masks = self._build_training_arrays(
            patches, progress_callback=patch_progress
        )
        print(f"[TRAINING] Generated {len(images)} total training patches")

        if progress_callback:
            progress_callback(0.2)  # 20% after patch generation

        dataset = PatchDataset(images, masks)
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        print(
            f"[TRAINING] Split: {len(train_dataset)} train, {len(val_dataset)} validation"
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        if progress_callback:
            progress_callback(0.25)  # 25% after data splitting

        print("[TRAINING] Initializing model...")
        model = self._initialise_model()
        print(f"[TRAINING] Model initialized on device: {self.device}")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        if progress_callback:
            progress_callback(0.3)  # 30% after model initialization

        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())

        print(
            f"[TRAINING] Starting training on {len(dataset)} patches (train {len(train_dataset)}, val {len(val_dataset)})"
        )
        print(
            f"[TRAINING] Training for {epochs} epochs with batch size {batch_size}, learning rate {learning_rate}"
        )
        print("[TRAINING] " + "=" * 60)

        if message_callback:
            message_callback(
                f"Starting training on {len(dataset)} patches (train {len(train_dataset)}, val {len(val_dataset)})"
            )

        # Phase 2: Training epochs (30-100% of progress)
        for epoch in range(epochs):
            train_loss = self._run_epoch(
                model, train_loader, criterion, optimizer, training=True
            )
            val_loss = self._run_epoch(
                model, val_loader, criterion, optimizer, training=False
            )

            if message_callback:
                message_callback(
                    f"Epoch {epoch + 1}/{epochs} - train loss: {train_loss:.4f}, val loss: {val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

            if progress_callback:
                # Map epoch progress (0-1) to overall progress (0.3-1.0)
                progress_callback(0.3 + 0.7 * ((epoch + 1) / epochs))

        print(
            f"[TRAINING] Training complete! Best validation loss: {best_val_loss:.4f}"
        )
        print("[TRAINING] Loading best model state and saving checkpoint...")

        if progress_callback:
            progress_callback(0.95)  # 95% before saving checkpoint

        model.load_state_dict(best_state)
        checkpoint_path = self._save_checkpoint(model)
        print(f"[TRAINING] Checkpoint saved to: {checkpoint_path}")
        self.model_manager.load_checkpoint(checkpoint_path)

        if progress_callback:
            progress_callback(1.0)  # 100% complete

        return str(checkpoint_path)

    def _build_training_arrays(
        self,
        patches: Iterable[CuratedPatch],
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        images_list: List[np.ndarray] = []
        masks_list: List[np.ndarray] = []

        patches_list = list(patches)
        print(
            f"[TRAINING] Generating augmented patches from {len(patches_list)} curated regions..."
        )

        for patch_idx, patch in enumerate(
            tqdm(
                patches_list, desc="Processing curated patches", unit="patch"
            ),
            1,
        ):
            img, msk = generate_augmented_patches(patch)
            images_list.append(img)
            masks_list.append(msk)
            if patch_idx == 1:
                print(
                    f"[TRAINING]   First patch: generated {len(img)} augmented patches (shape: {img.shape})"
                )

            if progress_callback:
                progress_callback(patch_idx / len(patches_list))

        print("[TRAINING] Concatenating all patches...")
        images = np.concatenate(images_list, axis=0)
        masks = np.concatenate(masks_list, axis=0)
        print(
            f"[TRAINING] Final arrays: images {images.shape}, masks {masks.shape}"
        )
        return images, masks

    def _initialise_model(self) -> UNet:
        if self.model_manager.model is not None:
            print("[TRAINING]   Using currently loaded model")
            model = copy.deepcopy(self.model_manager.model)
        else:
            print(
                "[TRAINING]   No model loaded, attempting to load default model..."
            )
            temp_manager = ModelManager()
            try:
                temp_manager.load_default_model()
                print("[TRAINING]   Default model loaded successfully")
                model = copy.deepcopy(temp_manager.model)
            except Exception as e:
                print(f"[TRAINING]   Failed to load default model: {e}")
                print("[TRAINING]   Creating new UNet from scratch")
                model = UNet()
        assert model is not None
        model.to(self.device)
        return model

    def _run_epoch(
        self,
        model: UNet,
        dataloader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        training: bool,
    ) -> float:
        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        batches = 0

        with torch.set_grad_enabled(training):
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                batches += 1

        return total_loss / max(batches, 1)

    def _save_checkpoint(self, model: UNet) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.models_dir / f"gfap_custom_{timestamp}.ckpt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        return checkpoint_path
