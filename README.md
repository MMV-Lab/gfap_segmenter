# gfap-segmenter
[![License BSD-3](https://img.shields.io/pypi/l/gfap-segmenter.svg?color=green)](https://github.com/MMV-Lab/gfap-segmenter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/gfap-segmenter.svg?color=green)](https://pypi.org/project/gfap-segmenter)
[![Python Version](https://img.shields.io/pypi/pyversions/gfap-segmenter.svg?color=green)](https://python.org)
[![tests](https://github.com/MMV-Lab/gfap-segmenter/workflows/tests/badge.svg)](https://github.com/MMV-Lab/gfap-segmenter/actions)
[![codecov](https://codecov.io/gh/MMV-Lab/gfap-segmenter/branch/main/graph/badge.svg)](https://codecov.io/gh/MMV-Lab/gfap-segmenter)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/gfap-segmenter)](https://napari-hub.org/plugins/gfap-segmenter)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A Bachelor Thesis project to allow segmentation of astrocytes on GFAP staining using machine learning in napari.

----------------------------------

## Features
- Load `.tif/.tiff/.czi` microscopy data via **bioio** and display them in napari.
- Run tiled U-Net inference that produces probability maps and binary masks.
- Curate training patches with ≥300×300 validation and immediate colour feedback.
- Synthesize 1000 augmented patches per curated region and train in the background with progress updates.
- Import/export checkpoints (including ONNX) and optionally bundle curated training data.

## Installation
Install `gfap-segmenter` via [pip]:

```
pip install gfap-segmenter
```

If napari (Qt backend included) is not yet installed, use:

```
pip install "gfap-segmenter[all]"
```

### GPU support

The plugin automatically uses GPU acceleration when available. GPU support significantly speeds up both model training and inference (typically 10-50x faster than CPU).

#### Installation

The plugin relies on PyTorch. By default, `pip install gfap-segmenter` installs the CPU-only version of PyTorch. To enable GPU acceleration, you need a CUDA-enabled PyTorch build.

**Important:** PyTorch bundles its own CUDA runtime, so you don't need to install the full CUDA toolkit separately. You only need NVIDIA GPU drivers (usually pre-installed on systems with NVIDIA GPUs).

**For NVIDIA GPUs (CUDA):**

**Option 1: Fresh installation (recommended)**

1. Install PyTorch with CUDA support from [PyTorch's official installation guide](https://pytorch.org/get-started/locally/):

   **CUDA 11.8:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

   **CUDA 12.4:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

   **CUDA 12.8:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

   **CUDA 13.0:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   ```

2. Then install `gfap-segmenter`:
   ```bash
   pip install gfap-segmenter
   ```

**Option 2: Upgrade existing CPU-only PyTorch installation**

If you already have `gfap-segmenter` installed with CPU-only PyTorch, you can upgrade to CUDA-enabled PyTorch:

1. Uninstall the CPU-only PyTorch:
   ```bash
   pip uninstall torch
   ```

2. Install CUDA-enabled PyTorch (choose the version matching your driver):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

3. The plugin will automatically use the CUDA-enabled PyTorch - no need to reinstall `gfap-segmenter`.

**Note:** You don't need to match the exact CUDA version. PyTorch includes its own CUDA runtime. As long as your NVIDIA driver is compatible (typically driver 11.0+ for CUDA 11.x, or 12.0+ for CUDA 12.x), you can use any recent PyTorch CUDA build. If unsure, use CUDA 12.1.

**For Apple Silicon (macOS):**

MPS (Metal Performance Shaders) support is automatically available if you have PyTorch 1.12+ installed. No additional steps required - the plugin will automatically use your Apple Silicon GPU.

#### Verification

To verify that GPU support is enabled:

1. **Check in the plugin UI**: The status bar at the bottom of the plugin widget shows the current device (e.g., "Device: CUDA (NVIDIA GeForce RTX 3090)" or "Device: CPU").

2. **Check via Python**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA device: {torch.cuda.get_device_name(0)}")
       print(f"CUDA version: {torch.version.cuda}")
   ```

3. **Check PyTorch CUDA support** (command line):
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
   ```

**Checking your NVIDIA driver version** (optional, for reference):

- **Using nvidia-smi** (if NVIDIA drivers are installed):
  ```bash
  nvidia-smi
  ```
  Look for the "CUDA Version" line (shows the maximum CUDA version your driver supports).

- **Using nvcc** (if CUDA toolkit is installed):
  ```bash
  nvcc --version
  ```

- **Using PyTorch** (after PyTorch is installed):
  ```bash
  python -c "import torch; print(torch.version.cuda)"
  ```

#### Troubleshooting

**Problem: Plugin shows "Device: CPU" even though you have a GPU**

- **PyTorch CPU-only build**: You may have installed the CPU-only version of PyTorch. Uninstall and reinstall with CUDA support:
  ```bash
  pip uninstall torch
  pip install torch --index-url https://download.pytorch.org/whl/cu121  # Adjust CUDA version
  ```

- **CUDA driver mismatch**: Your CUDA driver version may not match the PyTorch CUDA version. Check compatibility:
  ```bash
  nvidia-smi  # Shows driver version
  python -c "import torch; print(torch.version.cuda)"  # Shows PyTorch CUDA version
  ```
  The driver version should be >= the PyTorch CUDA version.

- **CUDA not detected**: Ensure your NVIDIA drivers are up to date and CUDA is properly installed.

**Problem: "CUDA out of memory" errors**

- Reduce batch size in training or prediction settings
- Use a smaller patch size
- Close other GPU-intensive applications

**Problem: Training/inference is still slow**

- Verify GPU is actually being used (check status bar or console output)
- Ensure you installed the correct CUDA version for your GPU
- Check GPU utilization: `nvidia-smi` should show high GPU usage during training

#### Performance Notes

- The plugin automatically uses GPU for both training and inference when available - no configuration needed. The current device is displayed at the bottom of the plugin.
- If no GPU is detected, the plugin automatically falls back to CPU execution (slower but functional).

### Development version
```
pip install git+https://github.com/MMV-Lab/gfap-segmenter.git
```

## Usage overview
1. **Prediction tab** – load the default checkpoint or import a custom one, choose an image layer, and run inference to create probability and mask layers.
2. **Curate Patches tab** – draw rectangles in the dedicated shapes layer. Patches ≥300×300 are tinted blue; smaller ones are red. Add valid regions to the training set.
3. **Train tab** – launch background training; progress and logs stream into the widget. Trained models are stored under `~/.gfap_segmenter/models`.
4. **Model IO tab** – export/import checkpoint bundles. Bundles can optionally include the curated training data for reproducibility.

## Contributing
Contributions are very welcome. Tests can be run with [tox]; please ensure coverage at least stays the same before submitting a pull request.

## License
Distributed under the terms of the [BSD-3] license, "gfap-segmenter" is free and open source software.

## Issues
If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[file an issue]: https://github.com/MMV-Lab/gfap-segmenter/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
