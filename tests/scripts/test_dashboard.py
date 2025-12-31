#!/usr/bin/env python3
"""Tests for the training dashboard server."""

import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.server import (  # noqa: E402
    app,
    discover_runs,
    read_metric_csv,
    read_all_metrics,
)


class TestDashboardServer(TestCase):
    """Test cases for dashboard server functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        app.config["LOGS_DIR"] = Path(self.temp_dir)
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_run(self, run_id: str, metrics: dict):
        """Create a test run with metric CSV files."""
        run_dir = Path(self.temp_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        for metric_name, values in metrics.items():
            csv_path = run_dir / f"{metric_name}.csv"
            with open(csv_path, "w") as f:
                f.write("step,value\n")
                for i, val in enumerate(values):
                    f.write(f"{i},{val}\n")

    def test_discover_runs_empty(self):
        """Test discovering runs when directory is empty."""
        runs = discover_runs()
        self.assertEqual(runs, [])

    def test_discover_runs_with_data(self):
        """Test discovering runs with metric data."""
        self.create_test_run("run1", {"loss": [1.0, 0.5, 0.25]})
        self.create_test_run("run2", {"loss": [0.8, 0.4], "accuracy": [0.5, 0.7]})

        runs = discover_runs()
        self.assertEqual(len(runs), 2)

        run_ids = {r["id"] for r in runs}
        self.assertIn("run1", run_ids)
        self.assertIn("run2", run_ids)

    def test_read_metric_csv(self):
        """Test reading a metric CSV file."""
        self.create_test_run("test_run", {"train_loss": [1.0, 0.5, 0.25, 0.1]})

        data = read_metric_csv("test_run", "train_loss")
        self.assertIsNotNone(data)
        self.assertEqual(data["steps"], [0, 1, 2, 3])
        self.assertEqual(data["values"], [1.0, 0.5, 0.25, 0.1])

    def test_read_metric_csv_not_found(self):
        """Test reading a non-existent metric."""
        data = read_metric_csv("nonexistent", "loss")
        self.assertIsNone(data)

    def test_read_all_metrics(self):
        """Test reading all metrics for a run."""
        self.create_test_run(
            "multi_metric",
            {
                "loss": [1.0, 0.5],
                "accuracy": [0.5, 0.8],
                "lr": [0.01, 0.001],
            },
        )

        metrics = read_all_metrics("multi_metric")
        self.assertEqual(len(metrics), 3)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("lr", metrics)

    def test_api_runs_empty(self):
        """Test /api/runs with no data."""
        response = self.client.get("/api/runs")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, [])

    def test_api_runs_with_data(self):
        """Test /api/runs with training runs."""
        self.create_test_run("api_run", {"loss": [1.0]})

        response = self.client.get("/api/runs")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "api_run")

    def test_api_run_metrics(self):
        """Test /api/run/<id>/metrics endpoint."""
        self.create_test_run("metrics_run", {"loss": [1.0, 0.5]})

        response = self.client.get("/api/run/metrics_run/metrics")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("loss", data)
        self.assertEqual(data["loss"]["values"], [1.0, 0.5])

    def test_api_run_not_found(self):
        """Test /api/run/<id> with non-existent run."""
        response = self.client.get("/api/run/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_api_compare(self):
        """Test /api/compare endpoint."""
        self.create_test_run("compare1", {"train_loss": [1.0, 0.5]})
        self.create_test_run("compare2", {"train_loss": [0.8, 0.3]})

        response = self.client.get("/api/compare?runs=compare1,compare2&metric=train_loss")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("compare1", data)
        self.assertIn("compare2", data)

    def test_index_page(self):
        """Test that the index page loads."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ML Odyssey", response.data)


if __name__ == "__main__":
    main()
