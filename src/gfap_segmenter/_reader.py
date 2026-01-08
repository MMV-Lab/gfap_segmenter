"""Reader utilities for the GFAP napari plugin.

This reader wraps :mod:`bioio` (when available) to support common microscopy
formats used in the GFAP workflow. We currently handle TIFF/TIF stacks and CZI
files and expose them to napari as ``image`` layers with sensible metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    from bioio import BioImage

    # Explicitly import CZI backend to ensure it registers
    try:
        import bioio_czi  # noqa: F401 - import for side effect (registration)
    except ImportError:
        pass  # Backend not available, will fail later with better error message
    BIOIO_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when bioio unavailable
    BioImage = None  # type: ignore[assignment]
    BIOIO_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from skimage import io as skio
except Exception as exc:  # pragma: no cover
    raise ImportError("scikit-image is required to load TIFF files") from exc


PathLike = Union[str, Path]
LayerData = Tuple[np.ndarray, dict, str]

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".czi"}


def napari_get_reader(path: Union[PathLike, Sequence[PathLike]]):
    """Return a callable napari reader when the path is supported.

    napari may pass either a single path or a list of paths. We check the first
    entry to decide whether we can read the file(s).
    """

    paths = _ensure_sequence(path)
    first_suffix = Path(paths[0]).suffix.lower()
    if first_suffix not in SUPPORTED_EXTENSIONS:
        return None

    return lambda p: _read_as_layers(_ensure_sequence(p))


def _ensure_sequence(path: Union[PathLike, Sequence[PathLike]]) -> List[str]:
    if isinstance(path, (str, Path)):
        return [str(path)]
    return [str(p) for p in path]


def _read_as_layers(paths: Iterable[PathLike]) -> List[LayerData]:
    layer_data: List[LayerData] = []
    for path in paths:
        path_obj = Path(path)
        array, meta = _load_image(path_obj)
        meta.setdefault("name", path_obj.stem)
        layer_data.append((array, meta, "image"))
    return layer_data


def _load_image(path: Path) -> Tuple[np.ndarray, dict]:
    suffix = path.suffix.lower()
    meta: dict = {}

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")

    if BIOIO_AVAILABLE:
        try:
            return _load_with_bioio(path)
        except ImportError:
            # Don't catch ImportError - let it propagate with helpful message
            # This happens when backend is missing
            raise
        except (
            Exception
        ) as exc:  # pragma: no cover - fallback for runtime issues
            # For CZI files, don't fall back to scikit-image (it can't read them)
            if suffix == ".czi":
                raise RuntimeError(
                    f"Failed to load CZI file with bioio: {exc}\n"
                    "This may indicate a problem with the bioio-czi backend."
                ) from exc
            # For other formats, we can try scikit-image as fallback
            meta["load_warning"] = (
                "Falling back to scikit-image; bioio failed with: " f"{exc}"
            )

    # Fallback loader for environments without bioio or on failure
    # Only use for non-CZI files
    if suffix == ".czi":
        raise ImportError(
            "CZI file support requires 'bioio' and 'bioio-czi'. "
            "Install with: pip install bioio bioio-czi"
        )

    data = skio.imread(str(path))
    data = np.asarray(data)
    if data.ndim > 2:
        data = np.squeeze(data)
    return data, meta


def _load_with_bioio(path: Path) -> Tuple[np.ndarray, dict]:
    # For CZI files, ensure backend is imported and dependencies are available
    if path.suffix.lower() == ".czi":
        try:
            import bioio_czi  # noqa: F401 - ensure backend is registered
        except ImportError:
            raise ImportError(
                "CZI file support requires 'bioio-czi'. Install with:\n"
                "  pip install bioio-czi"
            )
        # Check for underlying CZI library dependency
        try:
            import pylibczirw  # noqa: F401
        except ImportError:
            try:
                import aicspylibczi  # noqa: F401
            except ImportError:
                raise ImportError(
                    "bioio-czi requires either 'pylibczirw' or 'aicspylibczi'. "
                    "Install with:\n"
                    "  pip install pylibczirw"
                )

    try:
        bio_img = BioImage(path)
    except Exception as exc:
        error_msg = str(exc)
        if (
            "Could not find a backend" in error_msg
            or "backend" in error_msg.lower()
        ):
            if path.suffix.lower() == ".czi":
                raise ImportError(
                    f"Could not load CZI file: {path.name}\n"
                    "The bioio-czi backend may not be properly installed or registered.\n"
                    "Try reinstalling: pip install --force-reinstall bioio-czi"
                ) from exc
            raise ImportError(
                f"Could not load {path.suffix.upper()} file: {path.name}\n"
                f"Required backend not found. Error: {error_msg}"
            ) from exc
        raise

    data_array = bio_img.data
    axis_names: Sequence[str] = tuple(getattr(data_array, "dims", ()))
    data = np.asarray(data_array)

    # We expect images to be 2D+C, with all other axes (T/S/Z) being size 1 (singleton)
    # For non-singleton T/S/Z axes, take first slice only (don't preserve the dimension)
    # For singleton dimensions, keep them as-is (don't squeeze)
    if axis_names:
        axes_to_reduce = ["T", "S", "Z"]
        for axis in axes_to_reduce:
            if axis in axis_names:
                idx = axis_names.index(axis)
                if data.shape[idx] > 1:
                    # Take first slice for non-singleton axes (this reduces the dimension)
                    data = np.take(data, indices=0, axis=idx)
                    axis_names = [ax for ax in axis_names if ax != axis]
                # If size == 1, we keep it as-is (preserve singleton dimensions)
    else:
        # No axis names - if > 3D, take first slice of leading dimensions if > 1
        # This handles cases where we have T/S/Z axes but no axis names
        while data.ndim > 3:
            if data.shape[0] > 1:
                data = np.take(data, indices=0, axis=0)
            else:
                # Already singleton, preserve it
                break

    # Ensure Y/X are the last two axes following napari conventions
    data, axis_names = _reorder_axes(data, axis_names)

    # Infer channel axis if not present but data suggests channels
    if not axis_names and data.ndim == 3:
        # No axis info but 3D - if first dimension is small, likely channels
        if data.shape[0] <= 10:
            axis_names = ["C", "Y", "X"]
        else:
            # Otherwise might be Z-stack, but we expect 2D+C, so treat as channels
            axis_names = ["C", "Y", "X"]
    elif not axis_names and data.ndim > 3:
        # > 3D without axis info - infer structure
        # Assume first small dimension is channels
        if data.shape[0] <= 10:
            axis_names = ["C"] + ["?"] * (data.ndim - 3) + ["Y", "X"]
        else:
            # Can't infer reliably, but we expect 2D+C
            axis_names = ["C"] + ["?"] * (data.ndim - 3) + ["Y", "X"]

    # Set metadata based on detected/inferred structure
    meta: dict = {}
    channel_names_list = None
    if axis_names and "C" in axis_names:
        channel_axis = axis_names.index("C")
        meta["channel_axis"] = channel_axis
        try:
            channel_names_list = list(bio_img.channel_names)
            # Store channel_names in nested metadata dict only
            # (don't store at top-level as napari passes all top-level keys as kwargs)
            meta.setdefault("metadata", {})[
                "channel_names"
            ] = channel_names_list
        except Exception:  # pragma: no cover - optional metadata
            pass
    elif data.ndim == 3 and data.shape[0] <= 10:
        # Inferred channel axis (when axis_names was missing)
        meta["channel_axis"] = 0
        try:
            channel_names_list = list(bio_img.channel_names)
            # Store channel_names in nested metadata dict only
            # (don't store at top-level as napari passes all top-level keys as kwargs)
            meta.setdefault("metadata", {})[
                "channel_names"
            ] = channel_names_list
        except Exception:  # pragma: no cover - optional metadata
            pass

    return data.astype(np.float32, copy=False), meta


def _reduce_axes(
    data: np.ndarray,
    axes: List[str],
    axes_to_drop: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    for axis in axes_to_drop:
        if axis in axes and data.shape[axes.index(axis)] > 1:
            idx = axes.index(axis)
            data = np.take(data, indices=0, axis=idx)
            axes.pop(idx)
        elif axis in axes:
            idx = axes.index(axis)
            data = np.squeeze(data, axis=idx)
            axes.pop(idx)
    return data, axes


def _reorder_axes(
    data: np.ndarray, axes: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    axes = list(axes)
    if not axes:
        return data, axes

    target_order = [ax for ax in axes if ax not in ("Y", "X")]
    target_order.extend(ax for ax in ("Y", "X") if ax in axes)

    if target_order == axes:
        return data, axes

    permutation = [axes.index(ax) for ax in target_order]
    data = np.transpose(data, permutation)
    return data, target_order
