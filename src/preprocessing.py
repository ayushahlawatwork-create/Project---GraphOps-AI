"""Leakage-conscious feature engineering and chronological split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

TARGET_COLUMN = "total_network_traffic"
RESOURCE_TARGET_COLUMNS = ("cpu_usage_percent", "memory_usage_percent")
FEATURE_COLUMNS = (
    "cpu_usage_percent",
    "memory_usage_percent",
    "disk_io_rate",
    "network_rx_bytes",
    "network_tx_bytes",
    TARGET_COLUMN,
    "traffic_lag_1",
    "traffic_lag_3",
    "traffic_lag_6",
    "traffic_mean_3",
    "traffic_mean_12",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)
ScalerName = Literal["robust", "minmax"]


def utc_nanoseconds(values: object) -> np.ndarray:
    """Normalize timestamps to integer nanoseconds independent of pandas storage unit."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns").asi8.copy()


@dataclass(frozen=True)
class ChronologicalBoundaries:
    """UTC timestamp boundaries for non-overlapping chronological targets."""

    train_start_ns: int
    train_end_ns: int
    validation_start_ns: int
    validation_end_ns: int
    test_start_ns: int
    test_end_ns: int


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build target and causal per-machine lag/clock features."""
    required = {
        "timestamp",
        "machine_id",
        "cpu_usage_percent",
        "memory_usage_percent",
        "disk_io_rate",
        "network_rx_bytes",
        "network_tx_bytes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns for preprocessing: {missing}")

    result = frame.sort_values(["machine_id", "timestamp"], kind="stable").copy()
    for column in (
        "cpu_usage_percent",
        "memory_usage_percent",
        "disk_io_rate",
        "network_rx_bytes",
        "network_tx_bytes",
    ):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result[list(required - {"timestamp", "machine_id"})].to_numpy()).all():
        raise ValueError("Telemetry contains non-finite numeric values")

    result[TARGET_COLUMN] = result["network_rx_bytes"] + result["network_tx_bytes"]
    grouped = result.groupby("machine_id", sort=False)[TARGET_COLUMN]
    for lag in (1, 3, 6):
        result[f"traffic_lag_{lag}"] = grouped.shift(lag)
    for window in (3, 12):
        result[f"traffic_mean_{window}"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
    lag_columns = [column for column in FEATURE_COLUMNS if column.startswith("traffic_lag_")]
    for column in [*lag_columns, "traffic_mean_3", "traffic_mean_12"]:
        result[column] = result[column].fillna(result[TARGET_COLUMN])

    timestamps = pd.to_datetime(result["timestamp"], utc=True)
    fractional_hour = (
        timestamps.dt.hour
        + timestamps.dt.minute / 60.0
        + timestamps.dt.second / 3600.0
    )
    weekday = timestamps.dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24.0)
    result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)

    if not np.isfinite(result[list(FEATURE_COLUMNS)].to_numpy()).all():
        raise ValueError("Feature engineering produced non-finite values")
    return result.reset_index(drop=True)


def chronological_boundaries(
    frame: pd.DataFrame, train_ratio: float = 0.70, validation_ratio: float = 0.15
) -> ChronologicalBoundaries:
    """Split by global unique timestamps so machines share identical cutoffs."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1")

    times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True).unique()).sort_values()
    if len(times) < 3:
        raise ValueError("At least three distinct timestamps are required for splitting")
    train_count = max(1, int(len(times) * train_ratio))
    validation_end_count = max(train_count + 1, int(len(times) * (train_ratio + validation_ratio)))
    validation_end_count = min(validation_end_count, len(times) - 1)
    nanos = times.as_unit("ns").asi8
    return ChronologicalBoundaries(
        train_start_ns=int(nanos[0]),
        train_end_ns=int(nanos[train_count - 1]),
        validation_start_ns=int(nanos[train_count]),
        validation_end_ns=int(nanos[validation_end_count - 1]),
        test_start_ns=int(nanos[validation_end_count]),
        test_end_ns=int(nanos[-1]),
    )


class TelemetryPreprocessor:
    """Fit independent feature and target scalers on training rows only."""

    def __init__(self, scaler: ScalerName = "robust") -> None:
        if scaler not in ("robust", "minmax"):
            raise ValueError("scaler must be 'robust' or 'minmax'")
        self.scaler_name = scaler
        self.feature_scaler: RobustScaler | MinMaxScaler | None = None
        self.target_scaler: RobustScaler | MinMaxScaler | None = None
        self.resource_scalers: dict[str, RobustScaler | MinMaxScaler] = {}
        self.feature_columns = list(FEATURE_COLUMNS)
        self.fitted_training_rows: int | None = None
        self.training_end_ns: int | None = None

    def fit(self, training_frame: pd.DataFrame) -> "TelemetryPreprocessor":
        """Fit scaler statistics from the supplied training partition only."""
        if training_frame.empty:
            raise ValueError("Cannot fit preprocessing scalers on an empty training set")
        scaler_type = RobustScaler if self.scaler_name == "robust" else MinMaxScaler
        self.feature_scaler = scaler_type()
        self.target_scaler = scaler_type()
        self.resource_scalers = {
            column: scaler_type() for column in RESOURCE_TARGET_COLUMNS
        }
        self.feature_scaler.fit(training_frame[self.feature_columns])
        self.target_scaler.fit(training_frame[[TARGET_COLUMN]])
        for column, resource_scaler in self.resource_scalers.items():
            resource_scaler.fit(training_frame[[column]])
        self.fitted_training_rows = len(training_frame)
        self.training_end_ns = int(utc_nanoseconds(training_frame["timestamp"]).max())
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform features/target with already-fitted training scalers."""
        if self.feature_scaler is None or self.target_scaler is None:
            raise RuntimeError("Preprocessor must be fit before transform")
        features = self.feature_scaler.transform(frame[self.feature_columns]).astype(
            np.float32
        )
        target = self.target_scaler.transform(frame[[TARGET_COLUMN]]).astype(np.float32)
        return features, target[:, 0]

    def transform_resource_targets(self, frame: pd.DataFrame) -> np.ndarray:
        """Scale future CPU/memory labels with scalers fitted only on training rows."""
        if not self.resource_scalers:
            raise RuntimeError("Preprocessor must be fit before transform")
        return np.column_stack(
            [
                self.resource_scalers[column]
                .transform(frame[[column]])
                .astype(np.float32)[:, 0]
                for column in RESOURCE_TARGET_COLUMNS
            ]
        )

    def inverse_resource_target(self, column: str, values: np.ndarray) -> np.ndarray:
        """Convert a scaled CPU or memory target back to utilization percent."""
        if column not in self.resource_scalers:
            raise RuntimeError("Preprocessor must be fit before inverse_resource_target")
        values_array = np.asarray(values)
        return self.resource_scalers[column].inverse_transform(
            values_array.reshape(-1, 1)
        ).reshape(values_array.shape)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Convert scaled target values back to bytes per five-minute interval."""
        if self.target_scaler is None:
            raise RuntimeError("Preprocessor must be fit before inverse_target")
        return self.target_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(
            np.asarray(values).shape
        )
