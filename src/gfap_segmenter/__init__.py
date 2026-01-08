try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._reader import napari_get_reader
from ._widget import SegmenterWidget
from ._writer import (
    export_model_bundle,
    import_model_bundle,
    write_multiple,
    write_single_image,
)
from .background import BackgroundExecutor
from .data_utils import (
    CuratedPatch,
    PatchDataset,
    ensure_min_patch_size,
    extract_patch_from_layers,
    generate_augmented_patches,
)
from .model_manager import ModelManager, PredictionResult

__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "export_model_bundle",
    "import_model_bundle",
    "ModelManager",
    "PredictionResult",
    "BackgroundExecutor",
    "CuratedPatch",
    "PatchDataset",
    "ensure_min_patch_size",
    "extract_patch_from_layers",
    "generate_augmented_patches",
    "SegmenterWidget",
)
