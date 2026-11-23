"""Safe ingestion and validation for machine telemetry CSV files."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "timestamp",
    "machine_id",
    "cpu_usage_percent",
    "memory_usage_percent",
    "disk_io_rate",
    "network_rx_bytes",
    "network_tx_bytes",
)
NUMERIC_COLUMNS = REQUIRED_COLUMNS[2:]
DEFAULT_FREQUENCY = pd.Timedelta(minutes=5)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataQualityReport:
    """Counts that make input quality and sampling irregularities visible."""

    row_count: int
    machine_count: int
    timestamp_count: int
    duplicate_machine_timestamp_count: int
    missing_value_count: int
    invalid_numeric_count: int
    invalid_timestamp_count: int
    invalid_machine_id_count: int
    irregular_gap_count: int
    missing_interval_count: int
    sampling_frequency: str
    irregular_gaps_by_machine: dict[str, int]
    missing_intervals_by_machine: dict[str, int]


def inspect_sampling(
    frame: pd.DataFrame, expected_frequency: pd.Timedelta = DEFAULT_FREQUENCY
) -> DataQualityReport:
    """Report raw data-quality problems and cadence without changing the input."""
    irregular_by_machine: dict[str, int] = {}
    missing_by_machine: dict[str, int] = {}
    missing_value_count = int(frame.isna().sum().sum())

    if "timestamp" in frame:
        parsed_timestamps = pd.to_datetime(
            frame["timestamp"], utc=True, errors="coerce", format="mixed"
        )
        invalid_timestamp_mask = frame["timestamp"].notna() & parsed_timestamps.isna()
    else:
        parsed_timestamps = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        invalid_timestamp_mask = pd.Series(False, index=frame.index)

    if "machine_id" in frame:
        machine_ids = frame["machine_id"].astype("string").str.strip()
        invalid_machine_mask = machine_ids.isna() | machine_ids.eq("")
    else:
        machine_ids = pd.Series(pd.NA, index=frame.index, dtype="string")
        invalid_machine_mask = pd.Series(True, index=frame.index)

    numeric_columns = [column for column in NUMERIC_COLUMNS if column in frame]
    invalid_numeric_count = 0
    for column in numeric_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        raw_present = frame[column].notna()
        invalid = raw_present & converted.isna()
        finite = np.isfinite(converted.fillna(0).to_numpy(dtype=np.float64))
        invalid |= pd.Series(
            converted.notna().to_numpy() & ~finite, index=frame.index
        )
        if column in ("cpu_usage_percent", "memory_usage_percent"):
            invalid |= converted.notna() & ~converted.between(0, 100)
        elif column in ("disk_io_rate", "network_rx_bytes", "network_tx_bytes"):
            invalid |= converted.notna() & converted.lt(0)
        invalid_numeric_count += int(invalid.sum())

    valid_keys = (
        parsed_timestamps.notna()
        & ~invalid_machine_mask
    )
    valid_pairs = pd.DataFrame(
        {
            "machine_id": machine_ids.loc[valid_keys].astype(str),
            "timestamp": parsed_timestamps.loc[valid_keys],
        }
    )
    duplicate_count = int(valid_pairs.duplicated(["machine_id", "timestamp"]).sum())

    sampling_rows = pd.DataFrame(
        {
            "machine_id": machine_ids.loc[valid_keys].astype(str),
            "timestamp": parsed_timestamps.loc[valid_keys],
        }
    )
    for machine_id, machine_frame in sampling_rows.groupby("machine_id", sort=True):
        gaps = machine_frame["timestamp"].sort_values().diff().dropna()
        irregular = gaps.ne(expected_frequency)
        missing_intervals = (
            np.floor(gaps / expected_frequency).astype("int64").sub(1).clip(lower=0)
        )
        irregular_by_machine[str(machine_id)] = int(irregular.sum())
        missing_by_machine[str(machine_id)] = int(missing_intervals.sum())

    return DataQualityReport(
        row_count=len(frame),
        machine_count=int(machine_ids.nunique()),
        timestamp_count=int(parsed_timestamps.nunique()),
        duplicate_machine_timestamp_count=duplicate_count,
        missing_value_count=missing_value_count,
        invalid_numeric_count=invalid_numeric_count,
        invalid_timestamp_count=int(invalid_timestamp_mask.sum()),
        invalid_machine_id_count=int(invalid_machine_mask.sum()),
        irregular_gap_count=sum(irregular_by_machine.values()),
        missing_interval_count=sum(missing_by_machine.values()),
        sampling_frequency=str(expected_frequency),
        irregular_gaps_by_machine=irregular_by_machine,
        missing_intervals_by_machine=missing_by_machine,
    )


def load_telemetry(
    path: str | Path,
    *,
    expected_frequency: pd.Timedelta = DEFAULT_FREQUENCY,
    strict_frequency: bool = False,
) -> pd.DataFrame:
    """Load telemetry, fail on invalid rows, and sort by machine and time.

    Cadence gaps are reported as a warning by default because the original
    reference CSV is irregular. Set ``strict_frequency`` for regular panels.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Telemetry CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required telemetry columns: {missing_columns}")
    quality = inspect_sampling(frame, expected_frequency)
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()

    missing_timestamp_count = int(frame["timestamp"].isna().sum())
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    invalid_timestamps = quality.invalid_timestamp_count
    if invalid_timestamps or missing_timestamp_count:
        raise ValueError(
            f"Found {invalid_timestamps} invalid and {missing_timestamp_count} missing timestamps"
        )

    frame["machine_id"] = frame["machine_id"].astype("string").str.strip()
    invalid_machine_ids = frame["machine_id"].isna() | frame["machine_id"].eq("")
    if invalid_machine_ids.any():
        raise ValueError(
            f"Found {int(invalid_machine_ids.sum())} missing or empty machine IDs"
        )

    missing_numeric_count = int(frame.loc[:, NUMERIC_COLUMNS].isna().sum().sum())
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if missing_numeric_count:
        raise ValueError(
            f"Found {missing_numeric_count} missing numeric telemetry values"
        )

    invalid_numeric_count = int(
        (~np.isfinite(frame.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64))).sum()
    )
    if invalid_numeric_count:
        raise ValueError(
            f"Found {invalid_numeric_count} invalid or non-finite numeric values"
        )

    invalid_percent = ~frame["cpu_usage_percent"].between(0, 100) | ~frame[
        "memory_usage_percent"
    ].between(0, 100)
    if invalid_percent.any():
        raise ValueError(
            f"Found {int(invalid_percent.sum())} CPU/memory values outside [0, 100]"
        )
    nonnegative_columns = ("disk_io_rate", "network_rx_bytes", "network_tx_bytes")
    negative_counts = {
        column: int(frame[column].lt(0).sum())
        for column in nonnegative_columns
        if frame[column].lt(0).any()
    }
    if negative_counts:
        raise ValueError(f"Negative values in non-negative telemetry fields: {negative_counts}")

    duplicate_count = int(frame.duplicated(["machine_id", "timestamp"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate machine_id + timestamp rows"
        )

    frame = frame.sort_values(["machine_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    report = inspect_sampling(frame, expected_frequency)
    if report.irregular_gap_count:
        message = (
            f"Found {report.irregular_gap_count} irregular per-machine gaps "
            f"({report.missing_interval_count} missing intervals at "
            f"{expected_frequency} cadence)"
        )
        if strict_frequency:
            raise ValueError(message)
        LOGGER.warning(message)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Telemetry CSV to validate")
    parser.add_argument(
        "--strict-frequency",
        action="store_true",
        help="Fail if any machine has a gap other than the expected cadence",
    )
    parser.add_argument("--frequency", default="5min")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    telemetry = load_telemetry(
        args.csv,
        expected_frequency=pd.Timedelta(args.frequency),
        strict_frequency=args.strict_frequency,
    )
    print(json.dumps(asdict(inspect_sampling(telemetry, pd.Timedelta(args.frequency))), indent=2))


if __name__ == "__main__":
    main()
