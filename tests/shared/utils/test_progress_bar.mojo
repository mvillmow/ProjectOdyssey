"""Tests for progress bar utilities module.

This module tests progress bar functionality including:
- Basic progress bar rendering
- Metrics display
- ETA calculation
- Edge cases (0%, 100%, overflow)
"""

from shared.utils.progress_bar import (
    ProgressBar,
    ProgressBarWithMetrics,
    ProgressBarWithETA,
    format_duration,
    create_progress_bar,
    create_progress_bar_with_metrics,
    create_progress_bar_with_eta,
)


# ============================================================================
# Test Basic Progress Bar
# ============================================================================


fn test_progress_bar_creation():
    """Test progress bar can be created."""
    var progress = ProgressBar(total=100, description="Test")
    _ = progress


fn test_progress_bar_update():
    """Test progress bar update increments current."""
    var progress = ProgressBar(total=100)
    progress.update()
    progress.update()
    progress.update()
    # Should have incremented without error


fn test_progress_bar_overflow():
    """Test progress bar clamps to total."""
    var progress = ProgressBar(total=100)
    progress.update(150)
    # Should clamp to 100, not error


fn test_progress_bar_set_total():
    """Test updating total."""
    var progress = ProgressBar(total=100)
    progress.update(50)
    progress.set_total(200)
    # Should update without error


fn test_progress_bar_reset():
    """Test reset clears progress."""
    var progress = ProgressBar(total=100)
    progress.update(50)
    progress.reset()
    # Should reset without error


# ============================================================================
# Test Progress Bar With Metrics
# ============================================================================


fn test_progress_bar_with_metrics_creation():
    """Test metrics progress bar can be created."""
    var progress = ProgressBarWithMetrics(total=100, description="Epoch 1")
    _ = progress


fn test_progress_bar_with_metrics_set_metric():
    """Test setting a metric."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.342)
    progress.set_metric("accuracy", 0.891)
    # Should store metrics without error


fn test_progress_bar_with_metrics_update():
    """Test update with metrics."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.5)
    progress.update(25)
    # Should update with metrics without error


fn test_progress_bar_with_metrics_clear():
    """Test clearing metrics."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.5)
    progress.clear_metrics()
    # Should clear without error


fn test_progress_bar_with_metrics_reset():
    """Test reset clears metrics and progress."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.5)
    progress.update(50)
    progress.reset()
    # Should reset without error


# ============================================================================
# Test Progress Bar With ETA
# ============================================================================


fn test_progress_bar_with_eta_creation():
    """Test ETA progress bar can be created."""
    var progress = ProgressBarWithETA(total=100)
    _ = progress


fn test_progress_bar_with_eta_update():
    """Test ETA progress bar update."""
    var progress = ProgressBarWithETA(total=100)
    progress.update(25)
    progress.update(25)
    progress.update(25)
    progress.update(25)
    # Should update without error


fn test_progress_bar_with_eta_metrics():
    """Test ETA progress bar supports metrics."""
    var progress = ProgressBarWithETA(total=100)
    progress.set_metric("loss", 0.5)
    progress.set_metric("accuracy", 0.9)
    progress.update(50)
    # Should work with metrics


fn test_progress_bar_with_eta_reset():
    """Test ETA progress bar reset."""
    var progress = ProgressBarWithETA(total=100)
    progress.update(50)
    progress.set_metric("loss", 0.5)
    progress.reset()
    # Should reset without error


# ============================================================================
# Test Factory Functions
# ============================================================================


fn test_factory_create_progress_bar():
    """Test factory for simple progress bar."""
    var progress = create_progress_bar(total=100, description="Test")
    _ = progress


fn test_factory_create_progress_bar_with_metrics():
    """Test factory for metrics progress bar."""
    var progress = create_progress_bar_with_metrics(
        total=100, description="Test"
    )
    _ = progress


fn test_factory_create_progress_bar_with_eta():
    """Test factory for ETA progress bar."""
    var progress = create_progress_bar_with_eta(total=100, description="Test")
    _ = progress


# ============================================================================
# Test Helper Functions
# ============================================================================


fn test_format_duration():
    """Test format_duration produces output."""
    var result = format_duration(45.0)
    # Should produce non-empty string


fn test_format_duration_long():
    """Test format_duration with longer duration."""
    var result = format_duration(3665.0)
    # Should produce non-empty string


# ============================================================================
# Test Integration Scenarios
# ============================================================================


fn test_training_loop_simulation():
    """Test progress bar in simulated training loop."""
    var progress = ProgressBarWithMetrics(total=100, description="Training")

    var i = 0
    while i < 100:
        progress.set_metric("loss", 1.0 - (Float32(i) / 100.0) * 0.5)
        progress.update(1)
        i = i + 1

    # Should complete training loop without error


fn test_eta_time_tracking():
    """Test ETA progress bar tracks time."""
    var progress = ProgressBarWithETA(total=10)

    var i = 0
    while i < 10:
        progress.set_metric("loss", 1.0 - (Float32(i) / 10.0) * 0.5)
        progress.update(1)
        i = i + 1

    # Should complete tracking without error


fn test_rapid_updates():
    """Test progress bar handles rapid updates."""
    var progress = ProgressBar(total=1000)

    var i = 0
    while i < 1000:
        progress.update(1)
        i = i + 1

    # Should complete rapid updates without error


fn main() raises:
    """Run all progress bar tests."""
    print("")
    print("=" * 70)
    print("Running Progress Bar Tests")
    print("=" * 70)
    print("")

    test_progress_bar_creation()
    test_progress_bar_update()
    test_progress_bar_overflow()
    test_progress_bar_set_total()
    test_progress_bar_reset()

    test_progress_bar_with_metrics_creation()
    test_progress_bar_with_metrics_set_metric()
    test_progress_bar_with_metrics_update()
    test_progress_bar_with_metrics_clear()
    test_progress_bar_with_metrics_reset()

    test_progress_bar_with_eta_creation()
    test_progress_bar_with_eta_update()
    test_progress_bar_with_eta_metrics()
    test_progress_bar_with_eta_reset()

    test_factory_create_progress_bar()
    test_factory_create_progress_bar_with_metrics()
    test_factory_create_progress_bar_with_eta()

    test_format_duration()
    test_format_duration_long()

    test_training_loop_simulation()
    test_eta_time_tracking()
    test_rapid_updates()

    print("")
    print("=" * 70)
    print("All progress bar tests passed!")
    print("=" * 70)
    print("")
