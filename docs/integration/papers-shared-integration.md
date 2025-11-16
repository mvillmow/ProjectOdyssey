# Papers and Shared Library Integration Guide

This guide explains how papers/ and shared/ directories work together.

## Overview

- **papers/**: Individual paper implementations
- **shared/**: Reusable ML/AI components

## Integration Pattern

Papers import from shared:
```mojo
from shared.core import Layer, Module
from shared.training import Optimizer
from shared.data import Dataset
```

## Quick Start

See [quick-start-new-paper.md](quick-start-new-paper.md) for creating new papers.
