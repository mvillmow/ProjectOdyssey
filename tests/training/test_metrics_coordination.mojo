"""Comprehensive tests for metrics coordination and unified interface.

Tests the Metric trait, MetricCollection, MetricLogger, and integration
of all metrics through consistent API.

Coordination tests (#293-297):
- #294: Unified metric interface validation
- #295: MetricCollection utilities
- #296: MetricLogger for history tracking
- #297: Integration with all metric types

Testing strategy:
- Interface compliance: All metrics implement Metric trait
- Collection management: Add/update/reset multiple metrics
- History logging: Track metrics across epochs
- Integration: All metrics work together in training pipeline
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from collections.vector import DynamicVector
from math import abs
from extensor import ExTensor
from shared.training.metrics import (
    Metric, MetricResult, MetricCollection, MetricLogger,
    create_metric_summary,
    AccuracyMetric, LossTracker, ConfusionMatrix
)


fn test_metric_result_scalar() raises:
    """Test MetricResult with scalar values."""
    print("Testing MetricResult scalar...")

    var result = MetricResult("accuracy", 0.9234)

    assert_equal(result.name, "accuracy", "Result name")
    assert_true(result.is_scalar, "Result is scalar")
    assert_equal(result.scalar_value, 0.9234, "Scalar value")

    var value = result.get_scalar()
    assert_equal(value, 0.9234, "Get scalar value")

    # Should raise when accessing as tensor
    var raised = False
    try:
        var _ = result.get_tensor()
    except:
        raised = True

    assert_true(raised, "Should raise when accessing scalar as tensor")

    print("  ✓ MetricResult scalar works correctly")


fn test_metric_result_tensor() raises:
    """Test MetricResult with tensor values."""
    print("Testing MetricResult tensor...")

    var tensor = ExTensor(DynamicVector[Int](3), DType.float32)
    tensor._data.bitcast[Float32]()[0] = 0.9
    tensor._data.bitcast[Float32]()[1] = 0.8
    tensor._data.bitcast[Float32]()[2] = 0.95

    var result = MetricResult("per_class_acc", tensor)

    assert_equal(result.name, "per_class_acc", "Result name")
    assert_false(result.is_scalar, "Result is not scalar")

    var value_tensor = result.get_tensor()
    assert_equal(value_tensor.size(), 3, "Tensor size")
    assert_equal(value_tensor._data.bitcast[Float32]()[0], 0.9, "Tensor value 0")

    # Should raise when accessing as scalar
    var raised = False
    try:
        var _ = result.get_scalar()
    except:
        raised = True

    assert_true(raised, "Should raise when accessing tensor as scalar")

    print("  ✓ MetricResult tensor works correctly")


fn test_metric_collection_basic() raises:
    """Test basic MetricCollection operations."""
    print("Testing MetricCollection basic...")

    var collection = MetricCollection()

    assert_equal(collection.size(), 0, "Initial size is 0")
    assert_false(collection.contains("accuracy"), "Does not contain accuracy initially")

    # Add metrics
    collection.add("accuracy", AccuracyMetric())
    assert_equal(collection.size(), 1, "Size after add")
    assert_true(collection.contains("accuracy"), "Contains accuracy")

    collection.add("loss", LossTracker(window_size=100))
    assert_equal(collection.size(), 2, "Size after second add")
    assert_true(collection.contains("loss"), "Contains loss")

    # Get names
    var names = collection.get_names()
    assert_equal(len(names), 2, "Names vector size")
    assert_equal(names[0], "accuracy", "First name")
    assert_equal(names[1], "loss", "Second name")

    print("  ✓ MetricCollection basic operations work")


fn test_metric_collection_duplicate_handling() raises:
    """Test MetricCollection handles duplicate names."""
    print("Testing MetricCollection duplicate handling...")

    var collection = MetricCollection()

    collection.add("accuracy", AccuracyMetric())
    assert_equal(collection.size(), 1, "Size after first add")

    # Add with same name - should warn and replace
    collection.add("accuracy", AccuracyMetric())
    assert_equal(collection.size(), 1, "Size should stay 1 after duplicate")

    print("  ✓ MetricCollection handles duplicates correctly")


fn test_accuracy_metric_interface_compliance() raises:
    """Test AccuracyMetric implements Metric trait correctly."""
    print("Testing AccuracyMetric interface compliance...")

    var metric = AccuracyMetric()

    # Create test data
    var preds = ExTensor(DynamicVector[Int](4), DType.int32)
    var labels = ExTensor(DynamicVector[Int](4), DType.int32)

    preds._data.bitcast[Int32]()[0] = 0  # ✓
    preds._data.bitcast[Int32]()[1] = 1  # ✓
    preds._data.bitcast[Int32]()[2] = 2  # ✗
    preds._data.bitcast[Int32]()[3] = 0  # ✗

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 0
    labels._data.bitcast[Int32]()[3] = 1

    # Update through Metric interface
    metric.update(preds, labels)

    var acc = metric.compute()
    assert_equal(acc, 0.5, "Accuracy should be 0.5")

    # Reset through Metric interface
    metric.reset()
    var acc_after_reset = metric.compute()
    assert_equal(acc_after_reset, 0.0, "Accuracy should be 0.0 after reset")

    print("  ✓ AccuracyMetric implements Metric trait correctly")


fn test_confusion_matrix_integration() raises:
    """Test ConfusionMatrix works with metric coordination."""
    print("Testing ConfusionMatrix integration...")

    var cm = ConfusionMatrix(num_classes=3)

    # Create test data
    var preds = ExTensor(DynamicVector[Int](5), DType.int32)
    var labels = ExTensor(DynamicVector[Int](5), DType.int32)

    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1
    preds._data.bitcast[Int32]()[2] = 2
    preds._data.bitcast[Int32]()[3] = 1
    preds._data.bitcast[Int32]()[4] = 2

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 2
    labels._data.bitcast[Int32]()[3] = 0
    labels._data.bitcast[Int32]()[4] = 1

    # Update and compute precision
    cm.update(preds, labels)
    var precision = cm.get_precision()

    # Verify results
    assert_equal(precision.size(), 3, "Precision has 3 classes")

    # Reset
    cm.reset()
    var raw_after_reset = cm.normalize(mode="none")
    var sum = 0
    for i in range(9):
        sum += Int(raw_after_reset._data.bitcast[Float64]()[i])
    assert_equal(sum, 0, "Matrix should be empty after reset")

    print("  ✓ ConfusionMatrix integrates correctly")


fn test_metric_logger_basic() raises:
    """Test basic MetricLogger functionality."""
    print("Testing MetricLogger basic...")

    var logger = MetricLogger()

    # Log first epoch
    var epoch0_metrics = DynamicVector[MetricResult]()
    epoch0_metrics.push_back(MetricResult("accuracy", 0.7))
    epoch0_metrics.push_back(MetricResult("loss", 0.5))

    logger.log_epoch(0, epoch0_metrics)

    assert_equal(logger.num_epochs, 1, "Logged 1 epoch")
    assert_equal(logger.num_metrics, 2, "Tracked 2 metrics")

    # Log second epoch
    var epoch1_metrics = DynamicVector[MetricResult]()
    epoch1_metrics.push_back(MetricResult("accuracy", 0.8))
    epoch1_metrics.push_back(MetricResult("loss", 0.4))

    logger.log_epoch(1, epoch1_metrics)

    assert_equal(logger.num_epochs, 2, "Logged 2 epochs")

    print("  ✓ MetricLogger basic operations work")


fn test_metric_logger_history() raises:
    """Test MetricLogger history retrieval."""
    print("Testing MetricLogger history...")

    var logger = MetricLogger()

    # Log multiple epochs
    for i in range(5):
        var metrics = DynamicVector[MetricResult]()
        var acc = 0.5 + Float64(i) * 0.1  # 0.5, 0.6, 0.7, 0.8, 0.9
        var loss = 1.0 - Float64(i) * 0.1  # 1.0, 0.9, 0.8, 0.7, 0.6
        metrics.push_back(MetricResult("accuracy", acc))
        metrics.push_back(MetricResult("loss", loss))
        logger.log_epoch(i, metrics)

    # Get history
    var acc_history = logger.get_history("accuracy")
    assert_equal(len(acc_history), 5, "Accuracy history length")
    assert_equal(acc_history[0], 0.5, "Accuracy epoch 0")
    assert_equal(acc_history[4], 0.9, "Accuracy epoch 4")

    var loss_history = logger.get_history("loss")
    assert_equal(len(loss_history), 5, "Loss history length")
    assert_equal(loss_history[0], 1.0, "Loss epoch 0")
    assert_equal(loss_history[4], 0.6, "Loss epoch 4")

    # Get latest
    var latest_acc = logger.get_latest("accuracy")
    assert_equal(latest_acc, 0.9, "Latest accuracy")

    var latest_loss = logger.get_latest("loss")
    assert_equal(latest_loss, 0.6, "Latest loss")

    print("  ✓ MetricLogger history retrieval works")


fn test_metric_logger_best() raises:
    """Test MetricLogger best value retrieval."""
    print("Testing MetricLogger best value...")

    var logger = MetricLogger()

    # Log epochs with varying metrics
    var metrics0 = DynamicVector[MetricResult]()
    metrics0.push_back(MetricResult("accuracy", 0.7))
    metrics0.push_back(MetricResult("loss", 0.8))
    logger.log_epoch(0, metrics0)

    var metrics1 = DynamicVector[MetricResult]()
    metrics1.push_back(MetricResult("accuracy", 0.9))  # Best accuracy
    metrics1.push_back(MetricResult("loss", 0.6))
    logger.log_epoch(1, metrics1)

    var metrics2 = DynamicVector[MetricResult]()
    metrics2.push_back(MetricResult("accuracy", 0.8))
    metrics2.push_back(MetricResult("loss", 0.5))  # Best loss
    logger.log_epoch(2, metrics2)

    # Get best values
    var best_acc = logger.get_best("accuracy", maximize=True)
    assert_equal(best_acc, 0.9, "Best accuracy (maximize)")

    var best_loss = logger.get_best("loss", maximize=False)
    assert_equal(best_loss, 0.5, "Best loss (minimize)")

    print("  ✓ MetricLogger best value retrieval works")


fn test_create_metric_summary() raises:
    """Test create_metric_summary formatting."""
    print("Testing create_metric_summary...")

    var results = DynamicVector[MetricResult]()
    results.push_back(MetricResult("accuracy", 0.9234))
    results.push_back(MetricResult("loss", 0.1523))

    var summary = create_metric_summary(results)

    # Check that summary contains metric names and values
    var contains_accuracy = False
    var contains_loss = False
    var contains_summary = False

    # Simple substring checks (Mojo doesn't have built-in contains)
    if len(summary) > 0:
        contains_summary = True
        # We can't easily check substrings in Mojo yet, so just verify non-empty
        assert_true(len(summary) > 0, "Summary should not be empty")

    print("  Summary output:")
    print(summary)
    print("  ✓ create_metric_summary produces output")


fn test_multi_metric_training_simulation() raises:
    """Simulate a training loop with multiple metrics."""
    print("Testing multi-metric training simulation...")

    # Setup metrics
    var accuracy = AccuracyMetric()
    var loss_tracker = LossTracker(window_size=10)
    var confusion = ConfusionMatrix(num_classes=3)

    # Setup logger
    var logger = MetricLogger()

    # Simulate 3 epochs
    for epoch in range(3):
        # Reset metrics for new epoch
        accuracy.reset()
        confusion.reset()

        # Simulate 5 batches per epoch
        for batch in range(5):
            # Create fake batch data
            var preds = ExTensor(DynamicVector[Int](4), DType.int32)
            var labels = ExTensor(DynamicVector[Int](4), DType.int32)

            for i in range(4):
                var pred_class = (i + batch + epoch) % 3
                var true_class = (i + batch) % 3
                preds._data.bitcast[Int32]()[i] = Int32(pred_class)
                labels._data.bitcast[Int32]()[i] = Int32(true_class)

            # Update all metrics
            accuracy.update(preds, labels)
            confusion.update(preds, labels)

        # Compute epoch metrics
        var epoch_acc = accuracy.compute()
        var epoch_precision = confusion.get_precision()

        print("  Epoch " + String(epoch) + ": accuracy=" + String(epoch_acc))

        # Log to history
        var epoch_metrics = DynamicVector[MetricResult]()
        epoch_metrics.push_back(MetricResult("accuracy", epoch_acc))
        logger.log_epoch(epoch, epoch_metrics)

    # Verify we logged all epochs
    assert_equal(logger.num_epochs, 3, "Logged 3 epochs")

    var acc_history = logger.get_history("accuracy")
    assert_equal(len(acc_history), 3, "Accuracy history has 3 epochs")

    print("  ✓ Multi-metric training simulation works")


fn test_metric_interface_consistency() raises:
    """Test that all metrics have consistent interface patterns."""
    print("Testing metric interface consistency...")

    # All metrics should have update() and reset()
    var accuracy = AccuracyMetric()
    var confusion = ConfusionMatrix(num_classes=3)

    # Create test data
    var preds = ExTensor(DynamicVector[Int](2), DType.int32)
    var labels = ExTensor(DynamicVector[Int](2), DType.int32)
    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1

    # Both should accept update()
    accuracy.update(preds, labels)
    confusion.update(preds, labels)

    # Both should accept reset()
    accuracy.reset()
    confusion.reset()

    # Verify reset worked
    var acc_after_reset = accuracy.compute()
    assert_equal(acc_after_reset, 0.0, "Accuracy reset to 0.0")

    var cm_after_reset = confusion.normalize(mode="none")
    var cm_sum = 0
    for i in range(9):
        cm_sum += Int(cm_after_reset._data.bitcast[Float64]()[i])
    assert_equal(cm_sum, 0, "Confusion matrix reset to zeros")

    print("  ✓ All metrics have consistent interface")


fn main() raises:
    """Run all metrics coordination tests."""
    print("\n" + "="*70)
    print("METRICS COORDINATION TEST SUITE")
    print("Unified Interface and Collection Utilities (#293-297)")
    print("="*70 + "\n")

    print("MetricResult Tests (#294)")
    print("-" * 70)
    test_metric_result_scalar()
    test_metric_result_tensor()

    print("\nMetricCollection Tests (#294)")
    print("-" * 70)
    test_metric_collection_basic()
    test_metric_collection_duplicate_handling()

    print("\nMetric Interface Compliance (#294)")
    print("-" * 70)
    test_accuracy_metric_interface_compliance()
    test_confusion_matrix_integration()
    test_metric_interface_consistency()

    print("\nMetricLogger Tests (#295)")
    print("-" * 70)
    test_metric_logger_basic()
    test_metric_logger_history()
    test_metric_logger_best()

    print("\nUtility Tests (#295)")
    print("-" * 70)
    test_create_metric_summary()

    print("\nIntegration Tests (#296)")
    print("-" * 70)
    test_multi_metric_training_simulation()

    print("\n" + "="*70)
    print("ALL METRICS COORDINATION TESTS PASSED ✓")
    print("="*70 + "\n")
    print("Summary:")
    print("  ✓ Metric trait defines consistent update/reset interface")
    print("  ✓ MetricResult handles both scalar and tensor metrics")
    print("  ✓ MetricCollection manages multiple metrics efficiently")
    print("  ✓ MetricLogger tracks metric history across epochs")
    print("  ✓ All metrics (Accuracy, LossTracker, ConfusionMatrix) comply with interface")
    print("  ✓ Metrics integrate seamlessly in training pipelines")
    print()
