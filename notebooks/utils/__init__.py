"""Utility modules for ML Odyssey Jupyter notebooks.

This package provides utilities for:
- Bridging Mojo code execution via subprocess
- Tensor conversion between NumPy and binary formats
- Visualization of training curves and tensors
- Progress tracking during training
"""

from notebooks.utils.mojo_bridge import run_mojo_script, compile_mojo_binary
from notebooks.utils.tensor_utils import numpy_to_mojo_binary, mojo_binary_to_numpy
from notebooks.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    visualize_tensor,
)
from notebooks.utils.progress import TrainingProgressBar

__all__ = [
    "run_mojo_script",
    "compile_mojo_binary",
    "numpy_to_mojo_binary",
    "mojo_binary_to_numpy",
    "plot_training_curves",
    "plot_confusion_matrix",
    "visualize_tensor",
    "TrainingProgressBar",
]

__version__ = "0.1.0"
