#!/usr/bin/env python3
"""
Code generation tools for ML Odyssey.

This package provides generators for creating boilerplate Mojo code:
- Model generator
- Layer generator
- Dataset generator
- Training script generator
- Test generator

Usage:
    python -m scripts.generators.mojo_gen [command] [options]
"""

__version__ = "0.1.0"
__all__ = [
    "generate_model",
    "generate_layer",
    "generate_dataset",
    "generate_training_script",
    "generate_tests",
]
