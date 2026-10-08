"""Structured historical forecasts and integration-friendly prediction records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.sequence_builder import SequenceSet

HORIZON_MINUTES = (30, 60, 90, 120)
TRAFFIC_STATUS_PRIORITY = {"normal": 0, "elevated": 1, "burst": 2}
RESOURCE_STATUS_PRIORITY = {"normal": 0, "high": 1}
HISTORICAL_COLUMNS = (
    "timestamp",
    "forecast_timestamp",
    "machine_id",
    "horizon_minutes",
    "predicted_network_traffic",
    "actual_network_traffic",
    "prediction_error",
    "burst_status",
    "predicted_cpu_usage_percent",
    "actual_cpu_usage_percent",
    "predicted_memory_usage_percent",
    "actual_memory_usage_percent",
    "current_resource_pressure",
    "predicted_future_resource_pressure",
    "actual_future_resource_pressure",
)


@dataclass(frozen=True)
class TrafficThresholds:
    """Traffic and resource thresholds estimated only from training observations."""

    elevated_traffic: float
    burst_traffic: float
    cpu_pressure: float
    memory_pressure: float
    elevated_percentile: float = 0.85
    burst_percentile: float = 0.95
    pressure_percentile: float = 0.95


def fit_traffic_thresholds(
    training_frame: pd.DataFrame,
    *,
    elevated_percentile: float = 0.85,
    burst_percentile: float = 0.95,
    pressure_percentile: float = 0.95,
) -> TrafficThresholds:
    """Derive post-processing thresholds from training observations only."""
    if training_frame.empty:
        raise ValueError("Cannot estimate thresholds from an empty training set")
    if not 0 < elevated_percentile < burst_percentile < 1:
        raise ValueError("Require 0 < elevated_percentile < burst_percentile < 1")
    if not 0 < pressure_percentile < 1:
        raise ValueError("pressure_percentile must be between 0 and 1")
    traffic = training_frame["total_network_traffic"].to_numpy(dtype=np.float64)
    return TrafficThresholds(
        elevated_traffic=float(np.quantile(traffic, elevated_percentile)),
        burst_traffic=float(np.quantile(traffic, burst_percentile)),
        cpu_pressure=float(
            np.quantile(training_frame["cpu_usage_percent"], pressure_percentile)
        ),
        memory_pressure=float(
            np.quantile(training_frame["memory_usage_percent"], pressure_percentile)
        ),
        elevated_percentile=elevated_percentile,
        burst_percentile=burst_percentile,
        pressure_percentile=pressure_percentile,
    )


def classify_traffic(value: float, thresholds: TrafficThresholds) -> str:
    """Classify network demand using training-only traffic thresholds.

    The burst threshold is stricter than the elevated threshold, so a value above the
    burst cutoff is always reported as a burst, while values between the two cutoffs
    are kept as elevated rather than silently collapsed into normal.
    """
    value = float(value)
    if value >= thresholds.burst_traffic:
        return "burst"
    if value >= thresholds.elevated_traffic:
        return "elevated"
    return "normal"


def classify_resource_pressure(
    cpu_percent: float, memory_percent: float, thresholds: TrafficThresholds
) -> str:
    """Flag future resource pressure using the training-derived CPU/memory cutoffs."""
    cpu_percent = float(cpu_percent)
    memory_percent = float(memory_percent)
    if cpu_percent >= thresholds.cpu_pressure or memory_percent >= thresholds.memory_pressure:
        return "high"
    return "normal"


def build_historical_inference_dataset(
    sequences: SequenceSet,
    predictions: np.ndarray,
    thresholds: TrafficThresholds,
    source_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create one auditable row per origin, machine, and forecast horizon."""
    expected_shape = (len(sequences), len(sequences.horizon_steps), 3)
    if predictions.shape != expected_shape:
        raise ValueError(
            f"Predictions must have shape {expected_shape}, got {predictions.shape}"
        )
    if len(source_frame) != len(sequences.features):
        raise ValueError("source_frame must align row-for-row with sequence features")

    rows: list[dict[str, Any]] = []
    origin_positions = sequences.origin_positions
    actual_traffic = sequences.batch_actual_targets(np.arange(len(sequences)))
    actual_resources = sequences.batch_actual_resources(np.arange(len(sequences)))
    source_cpu = source_frame["cpu_usage_percent"].to_numpy(dtype=np.float64)
    source_memory = source_frame["memory_usage_percent"].to_numpy(dtype=np.float64)

    for sample_index, origin_position in enumerate(origin_positions):
        origin_time = pd.Timestamp(
            sequences.timestamps_ns[origin_position], unit="ns", tz="UTC"
        )
        machine_id = str(sequences.machine_ids[origin_position])
        current_pressure = classify_resource_pressure(
            source_cpu[origin_position],
            source_memory[origin_position],
            thresholds,
        )
        for horizon_index, step in enumerate(sequences.horizon_steps):
            predicted = float(predictions[sample_index, horizon_index, 0])
            actual = float(actual_traffic[sample_index, horizon_index])
            predicted_cpu = float(predictions[sample_index, horizon_index, 1])
            predicted_memory = float(predictions[sample_index, horizon_index, 2])
            actual_cpu = float(actual_resources[sample_index, horizon_index, 0])
            actual_memory = float(actual_resources[sample_index, horizon_index, 1])
            forecast_time = pd.Timestamp(
                sequences.timestamps_ns[
                    sequences.target_positions[sample_index, horizon_index]
                ],
                unit="ns",
                tz="UTC",
            )
            rows.append(
                {
                    "timestamp": origin_time,
                    "forecast_timestamp": forecast_time,
                    "machine_id": machine_id,
                    "horizon_minutes": int(step * 5),
                    "predicted_network_traffic": predicted,
                    "actual_network_traffic": actual,
                    "prediction_error": predicted - actual,
                    "burst_status": classify_traffic(predicted, thresholds),
                    "predicted_cpu_usage_percent": predicted_cpu,
                    "actual_cpu_usage_percent": actual_cpu,
                    "predicted_memory_usage_percent": predicted_memory,
                    "actual_memory_usage_percent": actual_memory,
                    "current_resource_pressure": current_pressure,
                    "predicted_future_resource_pressure": classify_resource_pressure(
                        predicted_cpu, predicted_memory, thresholds
                    ),
                    "actual_future_resource_pressure": classify_resource_pressure(
                        actual_cpu, actual_memory, thresholds
                    ),
                }
            )
    return pd.DataFrame(rows, columns=HISTORICAL_COLUMNS)


def save_historical_inference(
    records: pd.DataFrame, path: str | Path
) -> Path:
    """Persist historical predictions as a stable, dashboard-friendly CSV."""
    missing = sorted(set(HISTORICAL_COLUMNS) - set(records.columns))
    if missing:
        raise ValueError(f"Historical inference records missing columns: {missing}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records.loc[:, HISTORICAL_COLUMNS].to_csv(output_path, index=False)
    return output_path


def latest_prediction_payload(records: pd.DataFrame) -> list[dict[str, Any]]:
    """Return one direct-forecast object per machine for downstream modules."""
    if records.empty:
        return []
    payload: list[dict[str, Any]] = []
    ordered = records.sort_values(["machine_id", "timestamp", "horizon_minutes"])
    for machine_id, machine_rows in ordered.groupby("machine_id", sort=True):
        latest_origin = machine_rows["timestamp"].max()
        latest = machine_rows.loc[machine_rows["timestamp"].eq(latest_origin)]
        predictions = {
            str(int(row.horizon_minutes)): float(row.predicted_network_traffic)
            for row in latest.itertuples()
        }
        resource_predictions = {
            str(int(row.horizon_minutes)): {
                "cpu_usage_percent": float(row.predicted_cpu_usage_percent),
                "memory_usage_percent": float(row.predicted_memory_usage_percent),
            }
            for row in latest.itertuples()
        }
        statuses = latest["burst_status"].tolist()
        future_pressure = latest["predicted_future_resource_pressure"].tolist()
        payload.append(
            {
                "machine_id": str(machine_id),
                "timestamp": latest_origin.isoformat(),
                "predictions": predictions,
                "burst_status": max(
                    statuses,
                    key=lambda status: TRAFFIC_STATUS_PRIORITY[status],
                ),
                "resource_predictions": resource_predictions,
                "current_resource_pressure": str(
                    latest["current_resource_pressure"].iloc[0]
                ),
                "predicted_future_resource_pressure": max(
                    future_pressure, key=lambda status: RESOURCE_STATUS_PRIORITY[status]
                ),
            }
        )
    return payload


def save_prediction_payload(
    records: pd.DataFrame, path: str | Path
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(latest_prediction_payload(records), indent=2),
        encoding="utf-8",
    )
    return output_path


def save_thresholds(thresholds: TrafficThresholds, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(thresholds), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
