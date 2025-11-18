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

# Export metric implementations
from .accuracy import top1_accuracy, topk_accuracy, per_class_accuracy, AccuracyMetric

# Future exports (to be implemented):
# from .loss_tracker import LossTracker
# from .confusion import ConfusionMatrix
# from .precision import Precision
# from .recall import Recall

