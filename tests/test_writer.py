from pathlib import Path

import numpy as np

from gfap_segmenter import (
    export_model_bundle,
    import_model_bundle,
    write_multiple,
    write_single_image,
)


def test_write_single_image(tmp_path):
    array = (np.random.rand(16, 16) * 255).astype(np.uint8)
    output_path = tmp_path / "prediction.tif"
    result_paths = write_single_image(str(output_path), array, {})
    assert output_path.exists()
    assert result_paths == [str(output_path)]


def test_write_multiple_layers(tmp_path):
    array = np.zeros((16, 16), dtype=np.float32)
    meta = {"name": "mask", "gfap_export_suffix": ".npy"}
    output_path = tmp_path / "export.npy"
    result_paths = write_multiple(str(output_path), [(array, meta, "image")])
    assert Path(result_paths[0]).suffix == ".npy"
    assert Path(result_paths[0]).exists()


def test_model_bundle_roundtrip(tmp_path):
    model_file = tmp_path / "model.ckpt"
    model_file.write_bytes(b"dummy")

    bundle_path = tmp_path / "bundle.gfapmodel"
    bundle = export_model_bundle(
        bundle_path, [model_file], metadata={"epoch": 1}
    )
    assert Path(bundle).exists()

    restore_dir = tmp_path / "restore"
    info = import_model_bundle(bundle, restore_dir)
    restored_models = [Path(p) for p in info["models"]]
    assert restored_models and restored_models[0].exists()
    assert info["metadata"]["epoch"] == 1
