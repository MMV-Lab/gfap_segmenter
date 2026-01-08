"""GFAP Segmenter napari widget implementation."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
from qtpy.QtGui import QClipboard
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ._writer import export_model_bundle, import_model_bundle
from .background import BackgroundExecutor
from .data_utils import (
    CuratedPatch,
    ensure_min_patch_size,
    extract_patch_from_layers,
)
from .model_manager import ModelManager, PredictionResult

if TYPE_CHECKING:  # pragma: no cover - only used for typing
    import napari


VALID_PATCH_COLOR = [0.0, 0.4, 1.0, 0.25]
INVALID_PATCH_COLOR = [1.0, 0.1, 0.1, 0.25]


def show_error_dialog(
    parent: QWidget,
    title: str,
    error: Union[Exception, str],
    log_targets: List[QTextEdit] | None = None,
) -> None:
    """Show error with truncation, details dialog, copy button, and logging.

    Parameters
    ----------
    parent : QWidget
        Parent widget for the dialog
    title : str
        Dialog title
    error : Union[Exception, str]
        Exception object or error message string
    log_targets : Optional[List[QTextEdit]]
        Optional list of QTextEdit widgets to append error messages to
    """
    # Get full error text
    if isinstance(error, Exception):
        full_text = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        short_text = str(error)[:200]
    else:
        full_text = str(error)
        short_text = full_text[:200]

    # Log to targets
    if log_targets:
        for log in log_targets:
            if log is not None:
                log.append(f"ERROR: {full_text}")

    # Show truncated message with details button
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle(title)
    display_text = short_text + ("..." if len(full_text) > 200 else "")
    msg.setText(display_text)
    details_btn = msg.addButton("Show Details", QMessageBox.ActionRole)
    msg.addButton(QMessageBox.Ok)
    msg.exec()

    if msg.clickedButton() == details_btn:
        # Show full details dialog
        dialog = QDialog(parent)
        dialog.setWindowTitle(f"{title} - Full Details")
        dialog.resize(800, 600)
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(full_text)
        text_edit.setFontFamily("Courier")
        layout.addWidget(text_edit)
        btn_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(
                full_text, mode=QClipboard.Mode.Clipboard
            )
        )
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(copy_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.exec()


class SegmenterWidget(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer | None = None) -> None:
        super().__init__()
        if viewer is None:
            try:  # pragma: no cover - requires napari runtime
                from napari import current_viewer

                viewer = current_viewer()
            except Exception as exc:  # pragma: no cover - runtime only
                raise TypeError(
                    "GFAP Segmenter requires a napari viewer instance"
                ) from exc
        if viewer is None:  # pragma: no cover - runtime only
            raise TypeError("GFAP Segmenter requires an active napari viewer")
        self.viewer = viewer
        self.model_manager = ModelManager()
        self.executor = BackgroundExecutor()
        self.curated_patches: List[CuratedPatch] = []

        self.prediction_tab = PredictionTab(self)
        self.curation_tab = PatchCurationTab(self)
        self.training_tab = TrainingTab(self)
        self.model_tab = ModelIOTab(self)

        tabs = QTabWidget()
        tabs.addTab(self.prediction_tab, "Predict")
        tabs.addTab(self.curation_tab, "Curate Patches")
        tabs.addTab(self.training_tab, "Train")
        tabs.addTab(self.model_tab, "Model IO")

        layout = QVBoxLayout()
        layout.addWidget(tabs)

        # Status bar showing device information
        device_info = self.model_manager.get_device_info()
        device_type = device_info["type"]
        device_name = device_info["name"]

        if (
            device_type == "cuda"
            and device_info["available"] == "True"
            or device_type == "mps"
            and device_info["available"] == "True"
        ):
            status_text = f"Device: {device_type.upper()} ({device_name})"
        else:
            status_text = f"Device: {device_name}"

        self.status_bar = QLabel(status_text)
        self.status_bar.setStyleSheet(
            "color: gray; font-size: 9pt; padding: 2px;"
        )
        layout.addWidget(self.status_bar)

        self.setLayout(layout)

        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)
        self.viewer.layers.events.changed.connect(self._on_layers_changed)
        self._on_layers_changed()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def get_image_layers(self) -> List[napari.layers.Image]:
        return [
            layer
            for layer in self.viewer.layers
            if layer.__class__.__name__ == "Image"
        ]

    def get_labels_layers(self) -> List[napari.layers.Labels]:
        return [
            layer
            for layer in self.viewer.layers
            if layer.__class__.__name__ == "Labels"
        ]

    @staticmethod
    def extract_channel(
        layer: napari.layers.Image, channel_idx: int
    ) -> np.ndarray:
        """Extract a specific channel from a napari layer.

        For single-channel images, the image is treated as if the selected
        channel (the only channel) is used - the data is returned directly.

        Parameters
        ----------
        layer : napari.layers.Image
            The napari image layer
        channel_idx : int
            Index of the channel to extract (0-based).
            For single-channel images, this parameter is ignored.

        Returns
        -------
        np.ndarray
            2D array of the selected channel (or the image itself if single-channel)
        """
        data = np.asarray(layer.data)
        if data.ndim <= 2:
            return data

        # Get channel axis from metadata (may not be present)
        channel_axis = layer.metadata.get("channel_axis", None)
        # channel_names is stored directly in metadata (napari flattens nested structure)
        channel_names = layer.metadata.get("channel_names")

        # Determine if this is multi-channel and which axis contains channels
        is_multi_channel = False
        inferred_channel_axis = None
        num_channels = 0

        if channel_axis is not None and channel_axis < data.ndim:
            # Use channel_axis if available
            num_channels = data.shape[channel_axis]
            is_multi_channel = num_channels > 1
            inferred_channel_axis = channel_axis
        elif channel_names and len(channel_names) > 1:
            # channel_names exists with multiple entries - infer channel axis
            num_channels = len(channel_names)
            is_multi_channel = num_channels > 1
            # Assume channels are in first dimension if it matches the number of channels
            if data.ndim >= 3 and data.shape[0] == num_channels:
                inferred_channel_axis = 0
        elif data.ndim == 3 and data.shape[0] <= 10:
            # 3D data with small first dimension - likely channels
            num_channels = data.shape[0]
            is_multi_channel = num_channels > 1
            inferred_channel_axis = 0

        if is_multi_channel and inferred_channel_axis is not None:
            # Clamp channel_idx to valid range
            channel_idx = max(0, min(channel_idx, num_channels - 1))
            # Take the selected channel
            channel_data = np.take(
                data, channel_idx, axis=inferred_channel_axis
            )
        else:
            # Single channel or couldn't determine channel axis
            # Treat as if the selected channel (the only channel) is used
            channel_data = data

        # Squeeze all singleton dimensions to get 2D
        result = np.squeeze(channel_data)
        # Ensure we have a 2D array
        if result.ndim == 0:
            result = result.reshape(1, 1)
        elif result.ndim == 1:
            result = result.reshape(1, -1)
        return result

    def add_curated_patch(self, patch: CuratedPatch) -> None:
        self.curated_patches.append(patch)
        self.training_tab.update_curated_count(len(self.curated_patches))

    def clear_curated_patches(self) -> None:
        self.curated_patches.clear()
        self.training_tab.update_curated_count(0)

    def _on_layers_changed(self, _event=None) -> None:
        self.prediction_tab.refresh_layers()
        self.curation_tab.refresh_layers()
        self.training_tab.refresh_layers()


class PredictionTab(QWidget):
    def __init__(self, parent_widget: SegmenterWidget) -> None:
        super().__init__()
        self.parent_widget = parent_widget
        self.viewer = parent_widget.viewer
        self.model_manager = parent_widget.model_manager
        self.executor = parent_widget.executor

        layout = QVBoxLayout()
        form = QFormLayout()

        self.layer_combo = QComboBox()
        form.addRow("Input layer", self.layer_combo)

        self.channel_combo = QComboBox()
        form.addRow("Channel", self.channel_combo)

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(64, 1024)
        self.patch_size_spin.setSingleStep(32)
        self.patch_size_spin.setValue(256)
        form.addRow("Patch size", self.patch_size_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 256)
        self.overlap_spin.setValue(32)
        form.addRow("Overlap", self.overlap_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(4)
        form.addRow("Batch size", self.batch_spin)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.load_default_btn = QPushButton("Load default model")
        self.load_model_btn = QPushButton("Load model…")
        self.predict_btn = QPushButton("Run prediction")
        btn_row.addWidget(self.load_default_btn)
        btn_row.addWidget(self.load_model_btn)
        btn_row.addWidget(self.predict_btn)
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("Model not loaded")
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)

        layout.addStretch(1)
        self.setLayout(layout)

        self.load_default_btn.clicked.connect(self._load_default_model)
        self.load_model_btn.clicked.connect(self._load_custom_model)
        self.predict_btn.clicked.connect(self._run_prediction)
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)
        self.refresh_layers()

    def refresh_layers(self) -> None:
        current = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in self.parent_widget.get_image_layers():
            self.layer_combo.addItem(layer.name)
        index = self.layer_combo.findText(current)
        if index >= 0:
            self.layer_combo.setCurrentIndex(index)
        self.layer_combo.blockSignals(False)
        self._on_layer_changed()  # Update channel selection

    def _on_layer_changed(self) -> None:
        """Update channel selection when layer changes."""
        layer = self._selected_layer()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()

        if layer is None:
            self.channel_combo.setEnabled(False)
            self.channel_combo.blockSignals(False)
            return

        data = np.asarray(layer.data)
        # channel_names is stored directly in metadata (napari flattens nested structure)
        channel_names = layer.metadata.get("channel_names")
        channel_axis = layer.metadata.get("channel_axis", None)

        # Check if this is a multi-channel image
        # Match the logic in extract_channel for consistency
        # Single-channel images are treated as if the selected channel (the only channel) is used
        is_multi_channel = False
        num_channels = 0
        if channel_axis is not None and channel_axis < data.ndim:
            num_channels = data.shape[channel_axis]
            is_multi_channel = num_channels > 1
        elif channel_names and len(channel_names) > 1:
            num_channels = len(channel_names)
            is_multi_channel = num_channels > 1
        elif data.ndim == 3 and data.shape[0] <= 10:
            # 3D data with small first dimension - likely channels
            num_channels = data.shape[0]
            is_multi_channel = num_channels > 1

        if is_multi_channel:
            self.channel_combo.setEnabled(True)
            # Populate with channel names if available, otherwise use indices
            for i in range(num_channels):
                if channel_names and i < len(channel_names):
                    self.channel_combo.addItem(f"{i}: {channel_names[i]}")
                else:
                    self.channel_combo.addItem(f"Channel {i}")
        else:
            # Single channel - treated as if the selected channel (the only channel) is used
            self.channel_combo.setEnabled(False)
            self.channel_combo.addItem("Single channel")

        self.channel_combo.blockSignals(False)

    # ------------------------------------------------------------------
    def _load_default_model(self) -> None:
        try:
            self.model_manager.load_default_model()
        except Exception as exc:  # pragma: no cover - Qt error handling
            show_error_dialog(
                self,
                "Load model failed",
                exc,
                log_targets=[self.parent_widget.training_tab.log_output],
            )
            return
        self.status_label.setText("Loaded default model")

    def _load_custom_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select model checkpoint or ONNX file",
            str(Path.home()),
            "Model files (*.ckpt *.pt *.pth *.onnx)",
        )
        if not filename:
            return
        try:
            self.model_manager.load_model(Path(filename))
        except Exception as exc:
            show_error_dialog(
                self,
                "Load model failed",
                exc,
                log_targets=[self.parent_widget.training_tab.log_output],
            )
            return
        self.status_label.setText(f"Loaded model from {filename}")

    def _run_prediction(self) -> None:
        layer = self._selected_layer()
        if layer is None:
            QMessageBox.warning(
                self, "No layer", "Select an image layer to predict."
            )
            return

        # Extract selected channel
        channel_idx = self.channel_combo.currentIndex()
        image = self.parent_widget.extract_channel(layer, channel_idx)

        # Ensure image is 2D (defensive check)
        image = np.asarray(image)
        if image.ndim > 2:
            # Try to squeeze out extra dimensions
            image = np.squeeze(image)
        if image.ndim != 2:
            QMessageBox.warning(
                self,
                "Invalid image",
                f"Expected 2D image, but got {image.ndim}D array with shape {image.shape}. "
                "Please select a single-channel image layer.",
            )
            return

        patch_size = self.patch_size_spin.value()
        overlap = self.overlap_spin.value()
        batch_size = self.batch_spin.value()

        print(
            f"[WIDGET] Starting prediction: image shape={image.shape}, patch_size={patch_size}, overlap={overlap}, batch_size={batch_size}"
        )

        self.predict_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Running prediction…")

        runner = self.executor.submit(
            self.model_manager.predict,
            image,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
        )
        runner.signals.progress.connect(self._on_progress_update)
        runner.signals.result.connect(self._on_prediction_finished)
        runner.signals.error.connect(self._on_prediction_error)
        runner.signals.finished.connect(self._on_prediction_complete)

    def _selected_layer(self):
        name = self.layer_combo.currentText()
        if not name:
            return None
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def _on_progress_update(self, value: float) -> None:
        self.progress.setValue(int(value * 100))

    def _on_prediction_finished(self, result: PredictionResult) -> None:
        base_name = self.layer_combo.currentText() or "prediction"
        self.viewer.add_image(
            result.probability,
            name=f"{base_name}_probability",
            blending="additive",
            colormap="magenta",
        )
        self.viewer.add_labels(result.mask, name=f"{base_name}_mask")
        self.status_label.setText("Prediction completed")

    def _on_prediction_error(self, message: str) -> None:
        show_error_dialog(
            self,
            "Prediction failed",
            message,
            log_targets=[self.parent_widget.training_tab.log_output],
        )
        self.status_label.setText("Prediction failed")

    def _on_prediction_complete(self) -> None:
        self.predict_btn.setEnabled(True)


class PatchCurationTab(QWidget):
    def __init__(self, parent_widget: SegmenterWidget) -> None:
        super().__init__()
        self.parent_widget = parent_widget
        self.viewer = parent_widget.viewer
        self.shapes_layer = None

        layout = QVBoxLayout()

        form = QFormLayout()
        self.image_combo = QComboBox()
        self.image_channel_combo = QComboBox()
        self.labels_combo = QComboBox()
        form.addRow("Image layer", self.image_combo)
        form.addRow("Image channel", self.image_channel_combo)
        form.addRow("Labels layer", self.labels_combo)
        layout.addLayout(form)

        self.ensure_layer_btn = QPushButton("Create patch layer")
        self.ensure_layer_btn.clicked.connect(self._ensure_shapes_layer)
        layout.addWidget(self.ensure_layer_btn)

        self.feedback_label = QLabel(
            "Draw rectangles (≥ 300×300) in the patch layer"
        )
        layout.addWidget(self.feedback_label)

        self.add_patch_btn = QPushButton("Add selected patch")
        self.add_patch_btn.clicked.connect(self._commit_current_patch)
        layout.addWidget(self.add_patch_btn)

        self.patches_list = QListWidget()
        layout.addWidget(self.patches_list)

        clear_btn = QPushButton("Clear curated patches")
        clear_btn.clicked.connect(self._clear_curated)
        layout.addWidget(clear_btn)

        layout.addStretch(1)
        self.setLayout(layout)

        self.refresh_layers()

    def refresh_layers(self) -> None:
        current_img = self.image_combo.currentText()
        current_lbl = self.labels_combo.currentText()

        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for layer in self.parent_widget.get_image_layers():
            self.image_combo.addItem(layer.name)
        idx = self.image_combo.findText(current_img)
        if idx >= 0:
            self.image_combo.setCurrentIndex(idx)
        self.image_combo.blockSignals(False)

        self.labels_combo.blockSignals(True)
        self.labels_combo.clear()
        for layer in self.parent_widget.get_labels_layers():
            self.labels_combo.addItem(layer.name)
        idx = self.labels_combo.findText(current_lbl)
        if idx >= 0:
            self.labels_combo.setCurrentIndex(idx)
        self.labels_combo.blockSignals(False)

        # Update channel selection when layers refresh
        self._on_image_layer_changed()

    def _on_image_layer_changed(self) -> None:
        """Update channel selection when image layer changes."""
        layer_name = self.image_combo.currentText()
        if not layer_name:
            self.image_channel_combo.blockSignals(True)
            self.image_channel_combo.clear()
            self.image_channel_combo.setEnabled(False)
            self.image_channel_combo.blockSignals(False)
            return

        try:
            layer = self.viewer.layers[layer_name]
        except KeyError:
            return

        data = np.asarray(layer.data)
        # channel_names is stored directly in metadata (napari flattens nested structure)
        channel_names = layer.metadata.get("channel_names")
        channel_axis = layer.metadata.get("channel_axis", None)

        self.image_channel_combo.blockSignals(True)
        self.image_channel_combo.clear()

        # Check if this is a multi-channel image
        # Match the logic in extract_channel for consistency
        # Single-channel images are treated as if the selected channel (the only channel) is used
        is_multi_channel = False
        num_channels = 0
        if channel_axis is not None and channel_axis < data.ndim:
            num_channels = data.shape[channel_axis]
            is_multi_channel = num_channels > 1
        elif channel_names and len(channel_names) > 1:
            num_channels = len(channel_names)
            is_multi_channel = num_channels > 1
        elif data.ndim == 3 and data.shape[0] <= 10:
            # 3D data with small first dimension - likely channels
            num_channels = data.shape[0]
            is_multi_channel = num_channels > 1

        if is_multi_channel:
            self.image_channel_combo.setEnabled(True)
            # Populate with channel names if available, otherwise use indices
            for i in range(num_channels):
                if channel_names and i < len(channel_names):
                    self.image_channel_combo.addItem(
                        f"{i}: {channel_names[i]}"
                    )
                else:
                    self.image_channel_combo.addItem(f"Channel {i}")
        else:
            # Single channel - treated as if the selected channel (the only channel) is used
            self.image_channel_combo.setEnabled(False)
            self.image_channel_combo.addItem("Single channel")

        self.image_channel_combo.blockSignals(False)

    def _ensure_shapes_layer(self) -> None:
        if (
            self.shapes_layer is not None
            and self.shapes_layer in self.viewer.layers
        ):
            return
        try:
            layer = self.viewer.layers["GFAP Patches"]
        except KeyError:
            layer = self.viewer.add_shapes(
                name="GFAP Patches",
                ndim=2,
                edge_color="white",
                edge_width=2,
                face_color=VALID_PATCH_COLOR,
                opacity=0.4,
            )
        self.shapes_layer = layer
        layer.events.data.connect(self._update_patch_feedback)
        self._update_patch_feedback()

    def _current_layers(self) -> Tuple[np.ndarray | None, np.ndarray | None]:
        img_layer_name = self.image_combo.currentText()
        lbl_layer_name = self.labels_combo.currentText()
        img = None
        lbl = None
        if img_layer_name:
            try:
                img_layer = self.viewer.layers[img_layer_name]
                # Extract selected channel
                channel_idx = self.image_channel_combo.currentIndex()
                img = self.parent_widget.extract_channel(
                    img_layer, channel_idx
                )
            except KeyError:
                img = None
        if lbl_layer_name:
            try:
                lbl = np.asarray(self.viewer.layers[lbl_layer_name].data)
                # Ensure labels are 2D
                if lbl.ndim > 2:
                    lbl = np.squeeze(lbl)
            except KeyError:
                lbl = None
        return img, lbl

    def _update_patch_feedback(self, _event=None) -> None:
        if self.shapes_layer is None:
            return
        if len(self.shapes_layer.data) == 0:
            self.feedback_label.setText(
                "Draw rectangles (≥ 300×300) in the patch layer"
            )
            return

        face_colors = self.shapes_layer.face_color.copy()
        valid_any = False
        for index, shape in enumerate(self.shapes_layer.data):
            coords = np.asarray(shape)
            min_y, min_x = coords[:, 0].min(), coords[:, 1].min()
            max_y, max_x = coords[:, 0].max(), coords[:, 1].max()
            width = abs(max_x - min_x)
            height = abs(max_y - min_y)
            is_valid = ensure_min_patch_size(width, height)
            face_colors[index] = (
                VALID_PATCH_COLOR if is_valid else INVALID_PATCH_COLOR
            )
            if is_valid:
                valid_any = True
        with self.shapes_layer.events.face_color.blocker():
            self.shapes_layer.face_color = face_colors

        if valid_any:
            self.feedback_label.setText(
                "Valid patches highlighted in blue; invalid in red"
            )
        else:
            self.feedback_label.setText(
                "All patches too small – ensure ≥ 300×300"
            )

    def _commit_current_patch(self) -> None:
        indices = list(self.shapes_layer.selected_data)
        if self.shapes_layer is None or len(indices) == 0:
            QMessageBox.warning(
                self, "No patch selected", "Select a patch rectangle first."
            )
            return
        elif len(indices) > 1:
            QMessageBox.warning(
                self,
                "Multiple patches selected",
                "Select only one patch at a time.",
            )
            return
        index = indices[0]

        img_data, lbl_data = self._current_layers()
        if img_data is None or lbl_data is None:
            QMessageBox.warning(
                self, "Missing layers", "Select both image and labels layers."
            )
            return

        coords = np.asarray(self.shapes_layer.data[index])
        min_y, min_x = coords[:, 0].min(), coords[:, 1].min()
        max_y, max_x = coords[:, 0].max(), coords[:, 1].max()
        width = abs(max_x - min_x)
        height = abs(max_y - min_y)

        if not ensure_min_patch_size(width, height):
            QMessageBox.warning(
                self,
                "Patch too small",
                "Please resize patch to at least 300×300",
            )
            return

        top_left = (int(round(min_y)), int(round(min_x)))
        bottom_right = (int(round(max_y)), int(round(max_x)))

        patch = extract_patch_from_layers(
            img_data, lbl_data, top_left, bottom_right
        )
        self.parent_widget.add_curated_patch(patch)
        self.patches_list.addItem(
            f"Patch {len(self.parent_widget.curated_patches)}: {width:.0f}×{height:.0f}"
        )

    def _clear_curated(self) -> None:
        self.parent_widget.clear_curated_patches()
        self.patches_list.clear()


class TrainingTab(QWidget):
    def __init__(self, parent_widget: SegmenterWidget) -> None:
        super().__init__()
        self.parent_widget = parent_widget
        self.viewer = parent_widget.viewer
        self.executor = parent_widget.executor
        self.model_manager = parent_widget.model_manager
        self.training_runner = None

        layout = QVBoxLayout()

        self.info_label = QLabel("Curated patches: 0")
        layout.addWidget(self.info_label)

        form = QFormLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(50)
        form.addRow("Epochs", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(1e-5, 1e-1)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(1e-3)
        form.addRow("Learning rate", self.lr_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(8)
        form.addRow("Batch size", self.batch_spin)

        layout.addLayout(form)

        self.train_btn = QPushButton("Start training")
        self.train_btn.clicked.connect(self._start_training)
        layout.addWidget(self.train_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        layout.addStretch(1)
        self.setLayout(layout)

    def refresh_layers(self) -> None:
        pass

    def update_curated_count(self, count: int) -> None:
        self.info_label.setText(f"Curated patches: {count}")

    def _start_training(self) -> None:
        if not self.parent_widget.curated_patches:
            QMessageBox.warning(
                self,
                "No patches",
                "Curate at least one patch before training.",
            )
            return

        config = {
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "batch_size": self.batch_spin.value(),
        }

        self.train_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log_output.clear()

        runner = self.executor.submit(
            run_training,
            self.parent_widget.curated_patches,
            config,
            model_manager=self.model_manager,
        )
        runner.signals.progress.connect(self._on_training_progress)
        runner.signals.message.connect(self._on_training_message)
        runner.signals.result.connect(self._on_training_result)
        runner.signals.error.connect(self._on_training_error)
        runner.signals.finished.connect(self._on_training_finished)
        self.training_runner = runner

    def _on_training_progress(self, value: float) -> None:
        self.progress.setValue(int(value * 100))

    def _on_training_message(self, message: str) -> None:
        self.log_output.append(message)

    def _on_training_result(self, result_path: str) -> None:
        self.log_output.append(
            f"Training finished. Model saved to {result_path}"
        )
        self.parent_widget.model_tab.set_last_trained_model(result_path)

    def _on_training_error(self, message: str) -> None:
        show_error_dialog(
            self,
            "Training failed",
            message,
            log_targets=[self.log_output],
        )

    def _on_training_finished(self) -> None:
        self.train_btn.setEnabled(True)


class ModelIOTab(QWidget):
    def __init__(self, parent_widget: SegmenterWidget) -> None:
        super().__init__()
        self.parent_widget = parent_widget
        self.model_manager = parent_widget.model_manager
        self.executor = parent_widget.executor
        self.last_trained_model: str | None = None

        layout = QVBoxLayout()

        export_group = QGroupBox("Export model bundle")
        export_layout = QFormLayout()
        self.export_path_edit = QLineEdit()
        browse_export_btn = QPushButton("Browse…")
        browse_export_btn.clicked.connect(self._choose_export_path)
        export_layout.addRow(
            "Output path",
            self._with_button(self.export_path_edit, browse_export_btn),
        )

        self.include_training_checkbox = QCheckBox("Include training data")
        export_layout.addRow("Training data", self.include_training_checkbox)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        export_btn = QPushButton("Export bundle")
        export_btn.clicked.connect(self._export_bundle)
        layout.addWidget(export_btn)

        import_group = QGroupBox("Import model bundle")
        import_layout = QHBoxLayout()
        import_btn = QPushButton("Import bundle…")
        import_btn.clicked.connect(self._import_bundle)
        import_layout.addWidget(import_btn)
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)

        self.last_model_label = QLabel("Last trained model: none")
        layout.addWidget(self.last_model_label)

        layout.addStretch(1)
        self.setLayout(layout)

    def _with_button(self, widget: QWidget, button: QPushButton) -> QWidget:
        container = QWidget()
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(widget)
        hbox.addWidget(button)
        container.setLayout(hbox)
        return container

    def _choose_export_path(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select bundle path",
            str(Path.home() / "gfap_model.gfapmodel"),
            "GFAP Model (*.gfapmodel)",
        )
        if filename:
            self.export_path_edit.setText(filename)

    def _export_bundle(self) -> None:
        path_text = self.export_path_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "No path", "Specify an export path.")
            return

        model_files: List[str] = []
        if self.last_trained_model:
            model_files.append(self.last_trained_model)
        elif self.model_manager.model_path:
            model_files.append(str(self.model_manager.model_path))
        else:
            QMessageBox.warning(
                self, "No model", "Load or train a model before exporting."
            )
            return

        training_paths: List[str] = []
        temp_dir = None
        try:
            if (
                self.include_training_checkbox.isChecked()
                and self.parent_widget.curated_patches
            ):
                import tempfile

                temp_dir = tempfile.TemporaryDirectory()
                temp_path = Path(temp_dir.name)
                for idx, patch in enumerate(
                    self.parent_widget.curated_patches, start=1
                ):
                    patch_file = temp_path / f"training_patch_{idx}.npz"
                    np.savez_compressed(
                        patch_file,
                        image=patch.image,
                        labels=patch.labels,
                        origin=patch.origin,
                    )
                    training_paths.append(str(patch_file))

            bundle_path = export_model_bundle(
                path_text,
                model_files,
                metadata={"source": "gfap-segmenter"},
                training_data=training_paths,
                include_training_data=self.include_training_checkbox.isChecked(),
            )
        except Exception as exc:
            show_error_dialog(
                self,
                "Export failed",
                exc,
                log_targets=[self.parent_widget.training_tab.log_output],
            )
            return
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

        QMessageBox.information(
            self, "Export complete", f"Bundle written to {bundle_path}"
        )

    def _import_bundle(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select GFAP model bundle",
            str(Path.home()),
            "GFAP Model (*.gfapmodel)",
        )
        if not filename:
            return

        try:
            info = import_model_bundle(
                filename, Path.home() / "gfap_model_import"
            )
        except Exception as exc:
            show_error_dialog(
                self,
                "Import failed",
                exc,
                log_targets=[self.parent_widget.training_tab.log_output],
            )
            return

        if info["models"]:
            self.model_manager.load_model(Path(info["models"][0]))
            self.last_model_label.setText(
                f"Imported model: {info['models'][0]}"
            )
        else:
            self.last_model_label.setText("Bundle loaded (no models)")

    def set_last_trained_model(self, model_path: str) -> None:
        self.last_trained_model = model_path
        self.last_model_label.setText(f"Last trained model: {model_path}")


# ----------------------------------------------------------------------
# Training backend (Step 4 integration is implemented in run_training)
# ----------------------------------------------------------------------


def run_training(
    patches: List[CuratedPatch],
    config: dict,
    *,
    model_manager: ModelManager,
    progress_callback=None,
    message_callback=None,
) -> str:
    """Proxy function replaced by training integration in Step 4."""

    from .training import TrainingManager  # local import to avoid cycles

    manager = TrainingManager(model_manager)
    return manager.train_from_curated_patches(
        patches,
        epochs=config.get("epochs", 50),
        learning_rate=config.get("learning_rate", 1e-3),
        batch_size=config.get("batch_size", 8),
        progress_callback=progress_callback,
        message_callback=message_callback,
    )
