"""
Training Metrics

Metric implementations for tracking training and evaluation performance.

Includes:
- Accuracy (Classification accuracy - top-1, top-k, per-class)
- LossTracker (Loss tracking and averaging)
- ConfusionMatrix (Confusion matrix for classification)
- Precision (Precision metric)
- Recall (Recall metric)

All metrics implement the Metric trait for consistent interface.
"""

# Export base metric interface and utilities
from .base import (
    Metric,
    MetricResult,
    MetricCollection,
    MetricLogger,
    create_metric_summary,
)

# Export metric implementations
from .accuracy import (
    top1_accuracy,
    topk_accuracy,
    per_class_accuracy,
    AccuracyMetric,
)
from .loss_tracker import LossTracker, Statistics, ComponentTracker
from .confusion_matrix import ConfusionMatrix

# Consolidated evaluation utilities
from .evaluate import (
    evaluate_with_predict,
    evaluate_logits_batch,
    compute_accuracy_on_batch,
)

# Results printing utilities
from .results_printer import (
    print_evaluation_summary,
    print_per_class_accuracy,
    print_confusion_matrix,
    print_training_progress,
    print_training_summary,
)

# Future exports (to be implemented):
# from .precision import Precision
# from .recall import Recall
