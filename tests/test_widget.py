from qtpy.QtWidgets import QTabWidget

from gfap_segmenter._widget import SegmenterWidget


def test_segmenter_widget_initialises(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = SegmenterWidget(viewer)
    tab_widget = widget.findChild(QTabWidget)
    assert tab_widget is not None
    assert tab_widget.count() == 4
    assert [tab_widget.tabText(i) for i in range(tab_widget.count())] == [
        "Predict",
        "Curate Patches",
        "Train",
        "Model IO",
    ]
