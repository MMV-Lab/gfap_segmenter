"""Writer utilities for the GFAP napari plugin.

The writer supports two categories of outputs:

* Prediction exports: persisted as image or label files suitable for further
  analysis.
* Model bundles: zip archives containing model checkpoints (and optionally the
  curated training data used to produce them) for portability and reuse.

The helper functions `export_model_bundle` and `import_model_bundle` are
exposed for direct use from the plugin widget.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Union,
)
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

try:  # pragma: no cover - optional dependency for TIFF IO
    import tifffile

    _TIFF_AVAILABLE = True
except Exception:  # pragma: no cover
    tifffile = None  # type: ignore[assignment]
    _TIFF_AVAILABLE = False

try:  # pragma: no cover - fallback image writer
    from skimage import io as skio
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scikit-image is required for writing TIFF predictions"
    ) from exc

if TYPE_CHECKING:  # pragma: no cover - typing only
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


DEFAULT_IMAGE_SUFFIX = ".tif"
MODEL_BUNDLE_SUFFIX = ".gfapmodel"
MODEL_DIR_NAME = "models"
TRAINING_DATA_DIR_NAME = "training_data"
METADATA_FILENAME = "metadata.json"


def write_single_image(
    path: str, data: Any, meta: Mapping[str, Any] | None = None
) -> list[str]:
    """Persist a single napari layer to disk.

    Parameters
    ----------
    path : str
        Target path selected by the user.
    data : Any
        The layer's ``data`` payload.
    meta : Mapping[str, Any] | None
        Additional metadata associated with the layer.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(data)
    meta = meta or {}
    requested_suffix = meta.get("gfap_export_suffix")
    suffix = requested_suffix or target.suffix or DEFAULT_IMAGE_SUFFIX
    export_path = _ensure_suffix(target, suffix)

    _write_image(export_path, array, meta)
    return [str(export_path)]


def write_multiple(path: str, data: list[FullLayerData]) -> list[str]:
    """Persist multiple layers, deriving filenames from the base path."""

    base_path = Path(path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    written_paths: list[str] = []
    for index, (layer_data, meta, layer_type) in enumerate(data):
        suffix = (
            meta.get("gfap_export_suffix")
            or base_path.suffix
            or DEFAULT_IMAGE_SUFFIX
        )
        slug = _slugify(meta.get("name") or f"layer_{index}")
        filename = f"{base_path.stem or base_path.name}_{slug}{suffix}"
        export_path = base_path.with_name(filename)
        written_paths.extend(
            write_single_image(str(export_path), layer_data, meta)
        )

    return written_paths


def export_model_bundle(
    destination: Union[str, Path],
    model_files: Sequence[Union[str, Path]],
    *,
    metadata: Mapping[str, Any] | None = None,
    training_data: Sequence[Union[str, Path]] | None = None,
    include_training_data: bool = False,
) -> str:
    """Create a portable model bundle archive.

    Parameters
    ----------
    destination : Union[str, Path]
        Output path for the archive (.gfapmodel).
    model_files : Sequence[Union[str, Path]]
        Model artefacts to include (e.g., .ckpt, .onnx files).
    metadata : Mapping[str, Any], optional
        Additional metadata persisted as JSON (merged with defaults).
    training_data : Sequence[Union[str, Path]], optional
        Paths (files or directories) to curated training data.
    include_training_data : bool
        When ``True`` the bundle will include the supplied training data.

    Returns
    -------
    str
        Path to the created archive.
    """

    destination_path = Path(destination)
    if destination_path.suffix != MODEL_BUNDLE_SUFFIX:
        destination_path = destination_path.with_suffix(MODEL_BUNDLE_SUFFIX)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    model_paths = _validate_existing(model_files, "model")
    training_paths = _validate_existing(training_data or [], "training data")

    if include_training_data and not training_paths:
        raise ValueError(
            "Training data must be provided when include_training_data=True"
        )

    bundle_metadata: Dict[str, Any] = {
        "models": [path.name for path in model_paths],
        "include_training_data": bool(
            include_training_data and training_paths
        ),
    }
    if metadata:
        bundle_metadata.update(metadata)

    with ZipFile(
        destination_path, mode="w", compression=ZIP_DEFLATED
    ) as archive:
        for model_path in model_paths:
            archive.write(
                model_path, arcname=str(Path(MODEL_DIR_NAME) / model_path.name)
            )

        if include_training_data:
            for source_path in training_paths:
                _add_path_to_archive(
                    archive, source_path, TRAINING_DATA_DIR_NAME
                )

        archive.writestr(
            METADATA_FILENAME, json.dumps(bundle_metadata, indent=2)
        )

    return str(destination_path)


def import_model_bundle(
    bundle_path: Union[str, Path],
    destination_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Extract a previously exported model bundle.

    Returns a mapping containing the extracted paths and metadata.
    """

    bundle = Path(bundle_path)
    if not bundle.exists():
        raise FileNotFoundError(bundle)

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with ZipFile(bundle, mode="r") as archive:
        archive.extractall(destination)
        try:
            metadata = json.loads(archive.read(METADATA_FILENAME))
        except KeyError:
            metadata = {}

    models_dir = destination / MODEL_DIR_NAME
    training_dir = destination / TRAINING_DATA_DIR_NAME

    return {
        "bundle_path": str(bundle),
        "extracted_to": str(destination),
        "models": [str(p) for p in models_dir.glob("*") if p.is_file()],
        "training_data": [
            str(p) for p in training_dir.rglob("*") if p.is_file()
        ],
        "metadata": metadata,
    }


def _write_image(
    path: Path, array: np.ndarray, meta: Mapping[str, Any]
) -> None:
    if path.suffix.lower() == ".npy":
        np.save(path, array)
        return

    if array.dtype == bool:
        array = array.astype(np.uint8) * 255

    if _TIFF_AVAILABLE and path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(path, array, metadata=dict(meta))
        return

    skio.imsave(path, array, check_contrast=False)


def _ensure_suffix(path: Path, suffix: str) -> Path:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return path.with_suffix(suffix)


def _slugify(value: str) -> str:
    value = value.strip().replace(" ", "_")
    allowed = [c for c in value if c.isalnum() or c in {"_", "-"}]
    return "".join(allowed) or "layer"


def _validate_existing(
    paths: Sequence[Union[str, Path]], label: str
) -> List[Path]:
    resolved: List[Path] = []
    for candidate in paths:
        path = Path(candidate)
        if not path.exists():
            raise FileNotFoundError(
                f"{label.title()} path does not exist: {path}"
            )
        resolved.append(path)
    return resolved


def _add_path_to_archive(
    archive: ZipFile, source: Path, root_dir_name: str
) -> None:
    source = Path(source)
    if source.is_file():
        archive.write(source, arcname=str(Path(root_dir_name) / source.name))
        return

    for file_path in source.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(source)
            archive.write(
                file_path,
                arcname=str(Path(root_dir_name) / source.name / relative),
            )
