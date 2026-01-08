import numpy as np
import tifffile

from gfap_segmenter import napari_get_reader


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    my_test_file = tmp_path / "myfile.tif"
    original_data = (np.random.rand(20, 20) * 255).astype(np.uint8)
    tifffile.imwrite(my_test_file, original_data)

    # try to read it back in
    reader = napari_get_reader(str(my_test_file))
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(my_test_file))
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    data, meta, layer_type = layer_data_list[0]
    assert layer_type == "image"
    np.testing.assert_allclose(original_data, data.astype(original_data.dtype))
    assert meta["name"] == my_test_file.stem


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
