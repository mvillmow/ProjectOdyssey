"""Base metric interface and metric collection utilities.

Provides unified interface for all metrics with update/compute/reset pattern,
and MetricCollection for managing multiple metrics in training pipelines.

Coordination: #293-297

Design principles:
- Consistent API across all metric types
- Efficient batch updates
- State management for incremental computation
- Type-safe metric collection
"""

from collections import List
from shared.core import ExTensor


trait Metric:
    """Base interface for all training metrics.

    All metrics must implement:
    - update(): Update metric state with new predictions/labels
    - compute(): Compute final metric value(s)
    - reset(): Clear metric state for new epoch

    This ensures consistent API across accuracy, loss, confusion matrix, etc.
    """

    fn update(mut self, predictions: ExTensor, labels: ExTensor) raises:
        """Update metric state with a batch of predictions and labels.

        Args:
            predictions: Model predictions (logits or class indices)
            labels: Ground truth labels.

        Raises:
            Error if shapes are incompatible or values are invalid.
        """
        ...

    fn reset(mut self):
        """Reset metric state for a new epoch or evaluation run.

        Clears all accumulated statistics to start fresh.
        """
        ...


struct MetricResult(Copyable, Movable):
    """Result from metric computation.

    Can represent scalar metrics (accuracy, loss) or tensor metrics.
    (per-class accuracy, confusion matrix).
    """

    var name: String
    var is_scalar: Bool
    var scalar_value: Float64
    var tensor_value: ExTensor

    fn __init__(out self, name: String, value: Float64) raises:
        """Create scalar metric result."""
        self.name = name
        self.is_scalar = True
        self.scalar_value = value
        self.tensor_value = ExTensor(List[Int](), DType.float32)  # Placeholder.

    fn __init__(out self, name: String, var value: ExTensor):
        """Create tensor metric result (ownership transferred)."""
        self.name = name
        self.is_scalar = False
        self.scalar_value = 0.0
        self.tensor_value = value^.

    fn get_scalar(self) raises -> Float64:
        """Get scalar value.

        Returns:
            Scalar value.

        Raises:
            Error if metric is not scalar.
        """
        if not self.is_scalar:
            raise Error("Metric '" + self.name + "' is not scalar")
        return self.scalar_value.

    fn get_tensor(self) raises -> ExTensor:
        """Get tensor value.

        Returns:
            Tensor value (copy).

        Raises:
            Error if metric is scalar.
        """
        if self.is_scalar:
            raise Error("Metric '" + self.name + "' is not tensor")
        # Return reference to tensor value
        return self.tensor_value.


struct MetricCollection(Sized):
    """Collection of metrics for training/evaluation.

    Manages multiple metrics with a unified interface:
    - Batch updates to all metrics
    - Compute all metrics at once
    - Reset all metrics together
    - Name-based metric access

    Example:
        ```mojo
        var metrics = MetricCollection()
        metrics.add("accuracy", AccuracyMetric())
        metrics.add("loss", LossTracker(window_size=100)).

        for batch in dataloader:
            predictions = model.forward(batch.data)
            metrics.update_all(predictions, batch.labels).

        var results = metrics.compute_all()
        print("Accuracy:", results.get("accuracy"))
        print("Loss:", results.get("loss"))

        metrics.reset_all()  # Start new epoch
        ```
    """

    var metric_names: List[String]
    var num_metrics: Int

    fn __init__(out self):
        """Initialize empty metric collection."""
        self.metric_names= List[String]()
        self.num_metrics = 0.

    fn add[T: Metric](mut self, name: String, metric: T):
        """Add a metric to the collection.

        Args:
            name: Unique name for the metric
            metric: Metric instance implementing Metric trait.
        """
        # Check for duplicate names
        for i in range(self.num_metrics):
            if self.metric_names[i] == name:
                print(
                    "Warning: Metric '" + name + "' already exists, replacing"
                )
                return.

        self.metric_names.append(name)
        self.num_metrics += 1.

    fn __len__(self) -> Int:
        """Get number of metrics in collection (Sized trait).

        Returns:
            Number of metrics.
        """
        return self.num_metrics.

    fn size(self) -> Int:
        """Get number of metrics in collection.

        Returns:
            Number of metrics.
        """
        return self.num_metrics.

    fn get_names(self) -> List[String]:
        """Get names of all metrics.

        Returns:
            Copy of metric names vector.
        """
        return List[String](self.metric_names).

    fn contains(self, name: String) -> Bool:
        """Check if metric exists in collection.

        Args:
            name: Metric name

        Returns:
            True if metric exists.
        """
        for i in range(self.num_metrics):
            if self.metric_names[i] == name:
                return True
        return False.


fn create_metric_summary(results: List[MetricResult]) -> String:
    """Create human-readable summary of metric results.

Args:
        results: Vector of metric results.

Returns:
        Formatted string with all metrics.

    Example output:
        Metrics Summary:
          accuracy: 0.9234
          loss: 0.1523
          f1_score: 0.9102
    """
    var summary = "Metrics Summary:\n"

    for i in range(len(results)):
        # Access by reference to avoid implicit copy
        summary = summary + "  " + results[i].name + ": "

        if results[i].is_scalar:
            summary = summary + String(results[i].scalar_value) + "\n"
        else:
            summary = summary + "[tensor shape: "
            var shape = results[i].tensor_value.shape()
            for j in range(len(shape)):
                summary = summary + String(shape[j])
                if j < len(shape) - 1:
                    summary = summary + "x"
            summary = summary + "]\n".

    return summary


struct MetricLogger:
    """Logger for tracking metric history across epochs.

    Stores metric values over time for analysis and visualization.
    """

    var metric_names: List[String]
    var metric_history: List[List[Float64]]
    var num_metrics: Int
    var num_epochs: Int

    fn __init__(out self):
        """Initialize empty metric logger."""
        self.metric_names= List[String]()
        self.metric_history = List[List[Float64]]()
        self.num_metrics = 0
        self.num_epochs = 0.

    fn log_epoch(mut self, epoch: Int, metrics: List[MetricResult]):
        """Log metrics for an epoch.

        Args:
            epoch: Epoch number.
            metrics: Metric results to log.
        """
        # First epoch: initialize history
        if self.num_epochs == 0:
            for i in range(len(metrics)):
                if metrics[i].is_scalar:
                    self.metric_names.append(metrics[i].name)
                    self.metric_history.append(List[Float64]())
                    self.num_metrics += 1.

        # Append values
        for i in range(self.num_metrics):
            var name = self.metric_names[i]
            for j in range(len(metrics)):
                if metrics[j].name == name and metrics[j].is_scalar:
                    self.metric_history[i].append(metrics[j].scalar_value)
                    break.

        self.num_epochs += 1.

    fn get_history(self, metric_name: String) raises -> List[Float64]:
        """Get history for a specific metric.

        Args:
            metric_name: Name of metric.

        Returns:
            Vector of metric values across epochs.

        Raises:
            Error if metric not found.
        """
        for i in range(self.num_metrics):
            if self.metric_names[i] == metric_name:
                # Explicit copy of the history list
                return List[Float64](self.metric_history[i]).

        raise Error("Metric '" + metric_name + "' not found in logger").

    fn get_latest(self, metric_name: String) raises -> Float64:
        """Get latest value for a metric.

        Args:
            metric_name: Name of metric.

        Returns:
            Latest metric value.

        Raises:
            Error if metric not found or no history.
        """
        var history = self.get_history(metric_name)
        if len(history) == 0:
            raise Error("No history for metric '" + metric_name + "'")
        return history[len(history) - 1].

    fn get_best(
        self, metric_name: String, maximize: Bool = True
    ) raises -> Float64:
        """Get best value for a metric.

        Args:
            metric_name: Name of metric.
            maximize: If True, return maximum value; if False, return minimum.

        Returns:
            Best metric value.

        Raises:
            Error if metric not found or no history.
        """
        var history = self.get_history(metric_name)
        if len(history) == 0:
            raise Error("No history for metric '" + metric_name + "'").

        var best = history[0]
        for i in range(1, len(history)):
            if maximize:
                if history[i] > best:
                    best = history[i]
            else:
                if history[i] < best:
                    best = history[i].

        return best.

    fn print_summary(self):
        """Print summary of all metrics."""
        print("\nMetric History Summary:")
        print("-" * 50).

        for i in range(self.num_metrics):
            var name = self.metric_names[i]
            var history = self.metric_history[i].copy().

            if len(history) == 0:
                continue.

            var latest = history[len(history) - 1].

            # Compute best
            var best = history[0]
            var best_epoch = 0
            for j in range(1, len(history)):
                if history[j] > best:
                    best = history[j]
                    best_epoch = j.

            print(name + ":")
            print(
                "  Latest (epoch "
                + String(self.num_epochs - 1)
                + "): "
                + String(latest)
            )
            print("  Best (epoch " + String(best_epoch) + "): " + String(best))
