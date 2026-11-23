"""Machine-isolated sliding windows and chronological sequence partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing import ChronologicalBoundaries, utc_nanoseconds


@dataclass
class SequenceSet:
    """References into panel arrays; windows are materialized batch by batch."""

    features: np.ndarray
    scaled_target: np.ndarray
    raw_target: np.ndarray
    raw_resource_targets: np.ndarray
    timestamps_ns: np.ndarray
    machine_ids: np.ndarray
    origin_positions: np.ndarray
    target_positions: np.ndarray
    lookback: int
    horizon_steps: tuple[int, ...]
    scaled_resource_targets: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.origin_positions)

    @property
    def feature_count(self) -> int:
        return int(self.features.shape[1])

    def batch_inputs(self, sample_indices: np.ndarray) -> np.ndarray:
        """Return ``(batch, lookback, features)`` history windows."""
        positions = self.origin_positions[np.asarray(sample_indices, dtype=np.int64)]
        return np.stack(
            [
                self.features[position - self.lookback + 1 : position + 1]
                for position in positions
            ]
        ).astype(np.float32, copy=False)

    def batch_targets(self, sample_indices: np.ndarray) -> np.ndarray:
        positions = self.target_positions[np.asarray(sample_indices, dtype=np.int64)]
        traffic = self.scaled_target[positions].astype(np.float32, copy=False)
        if self.scaled_resource_targets is None:
            return traffic[:, :, None]
        resources = self.scaled_resource_targets[positions].astype(
            np.float32, copy=False
        )
        return np.concatenate((traffic[:, :, None], resources), axis=2)

    def batch_actual_targets(self, sample_indices: np.ndarray) -> np.ndarray:
        positions = self.target_positions[np.asarray(sample_indices, dtype=np.int64)]
        return self.raw_target[positions].astype(np.float64, copy=False)

    def batch_actual_resources(self, sample_indices: np.ndarray) -> np.ndarray:
        positions = self.target_positions[np.asarray(sample_indices, dtype=np.int64)]
        return self.raw_resource_targets[positions].astype(np.float64, copy=False)

    def select(self, mask: np.ndarray) -> "SequenceSet":
        mask = np.asarray(mask, dtype=bool)
        return SequenceSet(
            features=self.features,
            scaled_target=self.scaled_target,
            raw_target=self.raw_target,
            raw_resource_targets=self.raw_resource_targets,
            timestamps_ns=self.timestamps_ns,
            machine_ids=self.machine_ids,
            origin_positions=self.origin_positions[mask],
            target_positions=self.target_positions[mask],
            lookback=self.lookback,
            horizon_steps=self.horizon_steps,
            scaled_resource_targets=self.scaled_resource_targets,
        )


def build_sequences(
    frame: pd.DataFrame,
    features: np.ndarray,
    scaled_target: np.ndarray,
    *,
    lookback: int = 24,
    horizon_steps: tuple[int, ...] = (6, 12, 18, 24),
    scaled_resource_targets: np.ndarray | None = None,
) -> SequenceSet:
    """Create forecast origins separately within each machine's ordered history."""
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    if not horizon_steps or any(step < 1 for step in horizon_steps):
        raise ValueError("horizon_steps must contain positive step counts")
    if features.shape[0] != len(frame) or len(scaled_target) != len(frame):
        raise ValueError("Feature, target, and frame row counts must match")
    if scaled_resource_targets is not None and scaled_resource_targets.shape != (
        len(frame),
        2,
    ):
        raise ValueError("scaled_resource_targets must have shape (rows, 2)")

    ordered = frame.reset_index(drop=True)
    expected_order = ordered.sort_values(
        ["machine_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    if not ordered[["machine_id", "timestamp"]].equals(
        expected_order[["machine_id", "timestamp"]]
    ):
        raise ValueError(
            "frame, features, and targets must be aligned in machine/timestamp order"
        )
    timestamps_ns = utc_nanoseconds(ordered["timestamp"])
    machine_ids = ordered["machine_id"].astype(str).to_numpy()
    raw_target = ordered["total_network_traffic"].to_numpy(dtype=np.float64)
    raw_resource_targets = ordered[
        ["cpu_usage_percent", "memory_usage_percent"]
    ].to_numpy(dtype=np.float64)
    max_horizon = max(horizon_steps)
    origins: list[np.ndarray] = []

    for _, group in ordered.groupby("machine_id", sort=False):
        machine_positions = group.index.to_numpy(dtype=np.int64)
        if len(machine_positions) > 1 and not np.all(np.diff(machine_positions) == 1):
            raise ValueError("Each machine's rows must be contiguous in sequence input")
        if len(machine_positions) > 1 and not np.all(
            np.diff(timestamps_ns[machine_positions]) == pd.Timedelta(minutes=5).value
        ):
            raise ValueError(
                f"Machine {group['machine_id'].iloc[0]} does not have regular 5-minute cadence"
            )
        sample_count = len(machine_positions) - lookback - max_horizon + 1
        if sample_count <= 0:
            continue
        local_origins = np.arange(
            machine_positions[0] + lookback - 1,
            machine_positions[0] + lookback - 1 + sample_count,
            dtype=np.int64,
        )
        origins.append(local_origins)

    if not origins:
        raise ValueError("No sequences can be built with the requested window/horizons")
    origin_positions = np.concatenate(origins)
    target_positions = origin_positions[:, None] + np.asarray(horizon_steps)[None, :]
    return SequenceSet(
        features=features,
        scaled_target=np.asarray(scaled_target, dtype=np.float32),
        raw_target=raw_target,
        raw_resource_targets=raw_resource_targets,
        timestamps_ns=timestamps_ns,
        machine_ids=machine_ids,
        origin_positions=origin_positions,
        target_positions=target_positions,
        lookback=lookback,
        horizon_steps=horizon_steps,
        scaled_resource_targets=scaled_resource_targets,
    )


def chronological_partitions(
    sequences: SequenceSet, boundaries: ChronologicalBoundaries
) -> dict[str, SequenceSet]:
    """Select only sequences whose every label belongs to a single split."""
    target_times = sequences.timestamps_ns[sequences.target_positions]
    split_ranges = {
        "train": (boundaries.train_start_ns, boundaries.train_end_ns),
        "validation": (
            boundaries.validation_start_ns,
            boundaries.validation_end_ns,
        ),
        "test": (boundaries.test_start_ns, boundaries.test_end_ns),
    }
    partitions: dict[str, SequenceSet] = {}
    origin_times = sequences.timestamps_ns[sequences.origin_positions]
    for name, (start_ns, end_ns) in split_ranges.items():
        mask = (
            (origin_times >= start_ns)
            & (origin_times <= end_ns)
            & (target_times >= start_ns).all(axis=1)
            & (target_times <= end_ns).all(axis=1)
        )
        partitions[name] = sequences.select(mask)
        if not len(partitions[name]):
            raise ValueError(f"No forecast sequences available for {name} partition")
    return partitions
