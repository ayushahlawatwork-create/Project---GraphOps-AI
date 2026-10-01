from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.data_ingestion import inspect_sampling, load_telemetry
from src.generate_synthetic import generate_telemetry
from src.inference_engine import (
    HISTORICAL_COLUMNS,
    build_historical_inference_dataset,
    fit_traffic_thresholds,
    latest_prediction_payload,
)
from src.evaluate import _classification_metrics
from src.preprocessing import (
    FEATURE_COLUMNS,
    TelemetryPreprocessor,
    chronological_boundaries,
    engineer_features,
    utc_nanoseconds,
)
from src.sequence_builder import build_sequences, chronological_partitions
from src.traffic_forecaster import LSTMNetwork, TrafficForecaster, TrainingConfig


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "telemetry.csv"

    def write_rows(self, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(self.path, index=False)

    def rows(self) -> list[dict[str, object]]:
        base = {
            "machine_id": "node_1",
            "cpu_usage_percent": 20,
            "memory_usage_percent": 40,
            "disk_io_rate": 100,
            "network_rx_bytes": 200,
            "network_tx_bytes": 300,
        }
        return [
            {**base, "timestamp": "2026-01-01T00:05:00Z"},
            {**base, "timestamp": "2026-01-01T00:00:00Z"},
        ]

    def test_load_parses_utc_and_sorts_chronologically(self) -> None:
        self.write_rows(self.rows())
        frame = load_telemetry(self.path, strict_frequency=True)
        self.assertEqual(str(frame["timestamp"].dt.tz), "UTC")
        self.assertTrue(frame["timestamp"].is_monotonic_increasing)

    def test_schema_validation(self) -> None:
        pd.DataFrame([{"timestamp": "2026-01-01T00:00:00Z"}]).to_csv(
            self.path, index=False
        )
        with self.assertRaisesRegex(ValueError, "Missing required"):
            load_telemetry(self.path)

    def test_duplicate_detection(self) -> None:
        rows = self.rows()
        rows.append(dict(rows[0]))
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_telemetry(self.path)

    def test_missing_value_detection(self) -> None:
        rows = self.rows()
        rows[0]["cpu_usage_percent"] = None
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "missing numeric"):
            load_telemetry(self.path)

    def test_irregular_sampling_is_not_silently_accepted_as_regular(self) -> None:
        rows = self.rows()
        rows[1]["timestamp"] = "2026-01-01T00:00:00Z"
        rows[0]["timestamp"] = "2026-01-01T00:15:00Z"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "irregular"):
            load_telemetry(self.path, strict_frequency=True)

    def test_quality_report_counts_invalid_fields_separately(self) -> None:
        rows = self.rows()
        rows[0]["timestamp"] = "not-a-timestamp"
        rows[0]["network_rx_bytes"] = "not-numeric"
        rows[1]["cpu_usage_percent"] = None
        rows.append(dict(rows[1]))
        report = inspect_sampling(pd.DataFrame(rows))
        self.assertEqual(report.missing_value_count, 2)
        self.assertEqual(report.invalid_numeric_count, 1)
        self.assertEqual(report.invalid_timestamp_count, 1)
        self.assertEqual(report.duplicate_machine_timestamp_count, 1)
        self.assertGreaterEqual(report.irregular_gap_count, 0)

    def test_non_finite_and_out_of_range_telemetry_are_rejected(self) -> None:
        rows = self.rows()
        rows[0]["network_rx_bytes"] = float("inf")
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "non-finite numeric"):
            load_telemetry(self.path)

        rows = self.rows()
        rows[0]["cpu_usage_percent"] = 101
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "outside"):
            load_telemetry(self.path)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = generate_telemetry(machines=2, days=3, seed=12)
        cls.ingested = cls.frame.sort_values(
            ["machine_id", "timestamp"], kind="stable"
        ).reset_index(drop=True)
        cls.featured = engineer_features(cls.ingested)

    def test_synthetic_data_regular_panel(self) -> None:
        self.assertEqual(len(self.frame), 2 * 3 * 24 * 12)
        self.assertEqual(self.frame["machine_id"].nunique(), 2)
        self.assertEqual(
            int(self.frame.duplicated(["machine_id", "timestamp"]).sum()), 0
        )
        repeated = generate_telemetry(machines=2, days=3, seed=12)
        pd.testing.assert_frame_equal(self.frame, repeated)

    def test_sequences_have_shape_and_do_not_cross_machines(self) -> None:
        boundaries = chronological_boundaries(self.featured)
        train_frame = self.featured.loc[
            utc_nanoseconds(self.featured["timestamp"])
            <= boundaries.train_end_ns
        ]
        preprocessor = TelemetryPreprocessor()
        preprocessor.fit(train_frame)
        features, target = preprocessor.transform(self.featured)
        sequences = build_sequences(self.featured, features, target, lookback=24)
        inputs = sequences.batch_inputs(np.array([0, 1]))
        self.assertEqual(inputs.shape, (2, 24, len(FEATURE_COLUMNS)))
        for origin, targets in zip(
            sequences.origin_positions[:20], sequences.target_positions[:20]
        ):
            history_positions = np.arange(origin - 23, origin + 1)
            self.assertTrue(
                np.all(
                    sequences.machine_ids[history_positions]
                    == sequences.machine_ids[origin]
                )
            )
            self.assertTrue(
                np.all(sequences.machine_ids[targets] == sequences.machine_ids[origin])
            )

    def test_sequence_builder_rejects_unsorted_machine_time_rows(self) -> None:
        boundaries = chronological_boundaries(self.featured)
        train = self.featured.loc[
            utc_nanoseconds(self.featured["timestamp"]) <= boundaries.train_end_ns
        ]
        preprocessor = TelemetryPreprocessor().fit(train)
        features, target = preprocessor.transform(self.featured)
        reversed_frame = self.featured.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "machine/timestamp order"):
            build_sequences(
                reversed_frame,
                features,
                target,
                lookback=24,
            )

    def test_scalers_fit_training_rows_only(self) -> None:
        boundaries = chronological_boundaries(self.featured)
        times_ns = utc_nanoseconds(self.featured["timestamp"])
        training = self.featured.loc[times_ns <= boundaries.train_end_ns]
        later = self.featured.loc[times_ns > boundaries.train_end_ns]
        preprocessor = TelemetryPreprocessor("robust").fit(training)
        expected = RobustScaler().fit(training[list(FEATURE_COLUMNS)])
        self.assertEqual(preprocessor.fitted_training_rows, len(training))
        np.testing.assert_allclose(
            preprocessor.feature_scaler.center_, expected.center_
        )
        self.assertNotEqual(
            preprocessor.feature_scaler.center_[0],
            RobustScaler().fit(self.featured[list(FEATURE_COLUMNS)]).center_[0],
        )
        self.assertGreater(len(later), 0)
        minmax_preprocessor = TelemetryPreprocessor("minmax").fit(training)
        all_rows_with_future_extreme = self.featured.copy()
        all_rows_with_future_extreme.loc[
            times_ns > boundaries.train_end_ns, "cpu_usage_percent"
        ] = 1000
        self.assertEqual(
            tuple(minmax_preprocessor.feature_scaler.data_min_),
            tuple(MinMaxScaler().fit(training[list(FEATURE_COLUMNS)]).data_min_),
        )
        self.assertNotEqual(
            minmax_preprocessor.feature_scaler.data_max_[0],
            MinMaxScaler()
            .fit(all_rows_with_future_extreme[list(FEATURE_COLUMNS)])
            .data_max_[0],
        )
        for column in ("cpu_usage_percent", "memory_usage_percent"):
            np.testing.assert_allclose(
                preprocessor.resource_scalers[column].center_,
                RobustScaler().fit(training[[column]]).center_,
            )
        np.testing.assert_allclose(
            preprocessor.target_scaler.center_,
            RobustScaler().fit(training[["total_network_traffic"]]).center_,
        )

    def test_chronological_partitions_do_not_overlap_targets(self) -> None:
        boundaries = chronological_boundaries(self.featured)
        train = self.featured.loc[
            utc_nanoseconds(self.featured["timestamp"])
            <= boundaries.train_end_ns
        ]
        preprocessor = TelemetryPreprocessor().fit(train)
        features, target = preprocessor.transform(self.featured)
        sequences = build_sequences(self.featured, features, target, lookback=24)
        partitions = chronological_partitions(sequences, boundaries)
        split_ranges = {
            "train": (boundaries.train_start_ns, boundaries.train_end_ns),
            "validation": (
                boundaries.validation_start_ns,
                boundaries.validation_end_ns,
            ),
            "test": (boundaries.test_start_ns, boundaries.test_end_ns),
        }
        for name, partition in partitions.items():
            start_ns, end_ns = split_ranges[name]
            origin_times = sequences.timestamps_ns[partition.origin_positions]
            target_times = sequences.timestamps_ns[partition.target_positions]
            self.assertTrue(((origin_times >= start_ns) & (origin_times <= end_ns)).all())
            self.assertTrue(((target_times >= start_ns) & (target_times <= end_ns)).all())

    def test_24_step_production_test_origin_count(self) -> None:
        timestamps = pd.date_range(
            "2026-01-01", periods=30 * 24 * 12, freq="5min", tz="UTC"
        )
        index = pd.DatetimeIndex(timestamps)
        train_start = 0
        test_start = int(len(index) * 0.85)
        expected_per_machine = len(index) - 24 - test_start
        self.assertEqual(train_start, 0)
        self.assertEqual(expected_per_machine, 1272)
        self.assertEqual(expected_per_machine * 50, 63_600)

    def test_lstm_output_shape(self) -> None:
        model = LSTMNetwork(
            input_size=len(FEATURE_COLUMNS),
            output_size=12,
            horizon_count=4,
            target_count=3,
            hidden_size=8,
            num_layers=1,
            dropout=0.1,
        )
        result = model(torch.zeros((3, 24, len(FEATURE_COLUMNS))))
        self.assertEqual(tuple(result.shape), (3, 4, 3))

    def test_configured_seed_is_applied_before_model_initialization(self) -> None:
        config = TrainingConfig(seed=1780, device="cpu")
        first = TrafficForecaster(input_size=len(FEATURE_COLUMNS), config=config)
        first_weights = [value.detach().clone() for value in first.model.parameters()]
        second = TrafficForecaster(input_size=len(FEATURE_COLUMNS), config=config)
        for expected, observed in zip(first_weights, second.model.parameters()):
            torch.testing.assert_close(expected, observed, rtol=0, atol=0)

    def test_repeated_seeded_training_is_reproducible_on_cpu(self) -> None:
        boundaries = chronological_boundaries(self.featured)
        training_rows = self.featured.loc[
            utc_nanoseconds(self.featured["timestamp"]) <= boundaries.train_end_ns
        ]
        preprocessor = TelemetryPreprocessor().fit(training_rows)
        features, traffic = preprocessor.transform(self.featured)
        resources = preprocessor.transform_resource_targets(self.featured)
        sequences = build_sequences(
            self.featured,
            features,
            traffic,
            lookback=24,
            scaled_resource_targets=resources,
        )
        partitions = chronological_partitions(sequences, boundaries)
        config = TrainingConfig(
            seed=1780,
            device="cpu",
            epochs=2,
            patience=2,
            batch_size=256,
            hidden_size=8,
        )
        first = TrafficForecaster(len(FEATURE_COLUMNS), config=config)
        first.fit(partitions["train"], partitions["validation"])
        first_weights = [value.detach().clone() for value in first.model.parameters()]
        second = TrafficForecaster(len(FEATURE_COLUMNS), config=config)
        second.fit(partitions["train"], partitions["validation"])
        for expected, observed in zip(first_weights, second.model.parameters()):
            torch.testing.assert_close(expected, observed, rtol=0, atol=0)

    def test_thresholds_only_depend_on_training_rows(self) -> None:
        boundaries = chronological_boundaries(self.featured)
        train = self.featured.loc[
            utc_nanoseconds(self.featured["timestamp"]) <= boundaries.train_end_ns
        ].copy()
        thresholds = fit_traffic_thresholds(train)
        later = self.featured.loc[
            utc_nanoseconds(self.featured["timestamp"]) > boundaries.train_end_ns
        ].copy()
        later.loc[:, "total_network_traffic"] = 1e12
        later.loc[:, "cpu_usage_percent"] = 100
        later.loc[:, "memory_usage_percent"] = 100
        self.assertEqual(thresholds, fit_traffic_thresholds(train))
        self.assertNotEqual(
            thresholds,
            fit_traffic_thresholds(pd.concat([train, later], ignore_index=True)),
        )

    def test_burst_metrics_report_counts_and_positive_support(self) -> None:
        actual = np.array([[True, False], [True, True], [False, False]])
        predicted = np.array([[True, False], [False, True], [True, False]])
        rows = _classification_metrics(
            model_name="lstm",
            target_name="traffic_burst",
            horizon_steps=(6, 12),
            actual_positive=actual,
            predicted_positive=predicted,
        )
        self.assertEqual(rows[0]["support"], 2)
        self.assertEqual(rows[0]["true_positive"], 1)
        self.assertEqual(rows[0]["false_positive"], 1)
        self.assertEqual(rows[0]["false_negative"], 1)
        self.assertEqual(rows[0]["precision"], 0.5)
        self.assertAlmostEqual(rows[0]["f1"], 0.5)

    def test_resource_pressure_uses_future_predictions_not_current_state(self) -> None:
        boundaries = chronological_boundaries(self.featured)
        training = self.featured.loc[
            utc_nanoseconds(self.featured["timestamp"]) <= boundaries.train_end_ns
        ].copy()
        preprocessor = TelemetryPreprocessor().fit(training)
        features, target = preprocessor.transform(self.featured)
        sequences = build_sequences(
            self.featured,
            features,
            target,
            lookback=24,
            scaled_resource_targets=preprocessor.transform_resource_targets(
                self.featured
            ),
        )
        test = chronological_partitions(sequences, boundaries)["test"]
        thresholds = fit_traffic_thresholds(training)
        altered_sources = self.featured.copy()
        altered_sources.loc[test.origin_positions, "cpu_usage_percent"] = 0
        altered_sources.loc[test.origin_positions, "memory_usage_percent"] = 0
        predictions = np.zeros((len(test), 4, 3), dtype=np.float64)
        predictions[:, :, 1:] = 99.0
        records = build_historical_inference_dataset(
            test, predictions, thresholds, altered_sources
        )
        self.assertTrue((records["current_resource_pressure"] == "normal").all())
        self.assertTrue(
            (records["predicted_future_resource_pressure"] == "high").all()
        )

    def test_latest_forecast_payload_has_all_horizons_per_machine(self) -> None:
        boundaries = chronological_boundaries(self.featured)
        training = self.featured.loc[
            utc_nanoseconds(self.featured["timestamp"]) <= boundaries.train_end_ns
        ].copy()
        preprocessor = TelemetryPreprocessor().fit(training)
        features, target = preprocessor.transform(self.featured)
        sequences = build_sequences(
            self.featured,
            features,
            target,
            lookback=24,
            scaled_resource_targets=preprocessor.transform_resource_targets(
                self.featured
            ),
        )
        test = chronological_partitions(sequences, boundaries)["test"]
        thresholds = fit_traffic_thresholds(training)
        predictions = np.ones((len(test), 4, 3), dtype=np.float64)
        records = build_historical_inference_dataset(
            test, predictions, thresholds, self.featured
        )
        payload = latest_prediction_payload(records)
        self.assertEqual(len(payload), 2)
        self.assertEqual(
            {tuple(sorted(item["predictions"])) for item in payload},
            {("120", "30", "60", "90")},
        )
        self.assertEqual(
            {tuple(sorted(item["resource_predictions"])) for item in payload},
            {("120", "30", "60", "90")},
        )

    def test_inference_dataset_schema(self) -> None:
        boundaries = chronological_boundaries(self.featured)
        training = self.featured.loc[
            utc_nanoseconds(self.featured["timestamp"])
            <= boundaries.train_end_ns
        ]
        preprocessor = TelemetryPreprocessor().fit(training)
        features, target = preprocessor.transform(self.featured)
        sequences = build_sequences(
            self.featured,
            features,
            target,
            lookback=24,
            scaled_resource_targets=preprocessor.transform_resource_targets(
                self.featured
            ),
        )
        test = chronological_partitions(sequences, boundaries)["test"]
        thresholds = fit_traffic_thresholds(training)
        predictions = np.tile(
            training["total_network_traffic"].median(), (len(test), 4, 3)
        )
        predictions[:, :, 1] = training["cpu_usage_percent"].median()
        predictions[:, :, 2] = training["memory_usage_percent"].median()
        result = build_historical_inference_dataset(
            test, predictions, thresholds, self.featured
        )
        self.assertEqual(tuple(result.columns), HISTORICAL_COLUMNS)
        self.assertEqual(len(result), len(test) * 4)
        self.assertEqual(set(result["horizon_minutes"]), {30, 60, 90, 120})
        self.assertTrue(
            (result["forecast_timestamp"] > result["timestamp"]).all()
        )


if __name__ == "__main__":
    unittest.main()
