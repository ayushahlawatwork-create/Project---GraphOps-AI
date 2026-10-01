"""Generate deterministic, regular five-minute synthetic machine telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT = Path("data/synthetic/telemetry_forecasting.csv")
DEFAULT_START = "2026-01-01T00:00:00Z"
FREQUENCY = "5min"


def _ar1(rng: np.random.Generator, length: int, rho: float, noise_scale: float) -> np.ndarray:
    """Return a stationary, zero-centered AR(1) noise series."""
    innovations = rng.normal(0.0, noise_scale, length)
    series = np.empty(length, dtype=np.float64)
    series[0] = innovations[0]
    for index in range(1, length):
        series[index] = rho * series[index - 1] + innovations[index]
    return series


def _event_profile(
    rng: np.random.Generator, length: int, event_count: int, max_amplitude: float
) -> np.ndarray:
    """Create tapered, multi-sample workload events rather than isolated spikes."""
    profile = np.zeros(length, dtype=np.float64)
    for _ in range(event_count):
        start = int(rng.integers(0, max(1, length - 6)))
        duration = int(rng.integers(4, min(20, length - start) + 1))
        ramp = max(1, min(3, duration // 3))
        amplitude = float(rng.uniform(0.35, 1.0) * max_amplitude)
        for offset in range(duration):
            ramp_up = min(1.0, (offset + 1) / ramp)
            ramp_down = min(1.0, (duration - offset) / ramp)
            profile[start + offset] += amplitude * min(ramp_up, ramp_down)
    return profile


def _depletion_profile(
    rng: np.random.Generator, length: int, event_count: int
) -> np.ndarray:
    """Model slow memory pressure buildup and recovery over several hours."""
    profile = np.zeros(length, dtype=np.float64)
    for _ in range(event_count):
        duration = int(rng.integers(min(72, length), min(288, length) + 1))
        start = int(rng.integers(0, max(1, length - duration + 1)))
        rise = max(12, duration // 3)
        recovery = max(12, duration // 4)
        amplitude = float(rng.uniform(12.0, 21.0))
        for offset in range(duration):
            if offset < rise:
                fraction = (offset + 1) / rise
            elif offset >= duration - recovery:
                fraction = (duration - offset - 1) / recovery
            else:
                fraction = 1.0
            profile[start + offset] += amplitude * max(0.0, fraction)
    return profile


def generate_telemetry(
    *,
    machines: int = 50,
    days: int = 30,
    seed: int = 1780,
    start: str = DEFAULT_START,
) -> pd.DataFrame:
    """Generate a full machine-by-time panel with correlated temporal behavior."""
    if machines < 1:
        raise ValueError("machines must be at least 1")
    if days < 1:
        raise ValueError("days must be at least 1")

    rng = np.random.default_rng(seed)
    steps = days * 24 * 12
    timestamps = pd.date_range(start=start, periods=steps, freq=FREQUENCY, tz="UTC")
    step = np.arange(steps, dtype=np.float64)
    hour = step / 12.0
    day_of_week = (pd.DatetimeIndex(timestamps).dayofweek.to_numpy() + hour / 24.0)
    records: list[pd.DataFrame] = []

    for machine_index in range(machines):
        machine_id = f"node_{1001 + machine_index}"
        phase = float(rng.uniform(0, 2 * np.pi))
        base_workload = float(rng.uniform(32, 62))
        ar_noise = _ar1(rng, steps, rho=0.965, noise_scale=1.25)
        daily_cycle = 9.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0 + phase)
        weekly_cycle = 3.0 * np.sin(2 * np.pi * day_of_week / 7.0 + phase / 2)
        gradual_change = 4.0 * np.sin(2 * np.pi * step / (steps * 1.6) + phase)
        workload = np.clip(
            base_workload + daily_cycle + weekly_cycle + gradual_change + ar_noise,
            8,
            92,
        )

        event_count = max(3, days // 3)
        burst = _event_profile(rng, steps, event_count, max_amplitude=3.2)
        burst_workload = np.clip(workload + burst * 2.1, 0, 99)
        cpu_noise = _ar1(rng, steps, rho=0.72, noise_scale=1.8)
        cpu = np.clip(burst_workload + cpu_noise, 1, 99)

        memory_noise = _ar1(rng, steps, rho=0.992, noise_scale=0.12)
        memory = (
            float(rng.uniform(38, 68))
            + 0.16 * (workload - 48)
            + 2.4 * np.sin(2 * np.pi * hour / 24.0 + phase / 3)
            + memory_noise
        )
        pressure = _depletion_profile(rng, steps, max(1, days // 12))
        memory = np.clip(memory + pressure, 8, 99.5)

        common_demand = np.clip(0.55 * workload + 0.45 * cpu, 5, 100)
        traffic_noise = _ar1(rng, steps, rho=0.68, noise_scale=0.09)
        multiplier = np.exp(traffic_noise + rng.normal(0, 0.08, steps))
        rx_base = 1_050_000.0 * (0.55 + common_demand / 55.0)
        tx_base = 780_000.0 * (0.55 + common_demand / 58.0)
        rx = np.maximum(1.0, rx_base * multiplier + burst * 1_100_000.0)
        tx = np.maximum(
            1.0,
            tx_base * np.exp(0.72 * traffic_noise + rng.normal(0, 0.1, steps))
            + burst * 760_000.0,
        )
        disk_noise = np.exp(rng.normal(0, 0.28, steps))
        disk_io = np.maximum(
            1.0,
            130.0 + 34.0 * cpu + 21.0 * burst + 175.0 * disk_noise,
        )

        records.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "machine_id": machine_id,
                    "cpu_usage_percent": cpu.round(2),
                    "memory_usage_percent": memory.round(2),
                    "disk_io_rate": disk_io.round(2),
                    "network_rx_bytes": rx.round(2),
                    "network_tx_bytes": tx.round(2),
                }
            )
        )

    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["timestamp", "machine_id"], kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--machines", type=int, default=50)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1780)
    parser.add_argument("--start", default=DEFAULT_START)
    args = parser.parse_args()

    frame = generate_telemetry(
        machines=args.machines,
        days=args.days,
        seed=args.seed,
        start=args.start,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(
        f"Wrote {len(frame):,} rows for {frame['machine_id'].nunique()} machines "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
