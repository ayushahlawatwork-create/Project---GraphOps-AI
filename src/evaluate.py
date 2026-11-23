"""End-to-end chronological baseline/LSTM training, inference, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_ingestion import load_telemetry
from src.inference_engine import (
    HORIZON_MINUTES,
    build_historical_inference_dataset,
    fit_traffic_thresholds,
    save_historical_inference,
    save_prediction_payload,
    save_thresholds,
)
from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TelemetryPreprocessor,
    chronological_boundaries,
    engineer_features,
    utc_nanoseconds,
)
from src.sequence_builder import (
    SequenceSet,
    build_sequences,
    chronological_partitions,
)
from src.traffic_forecaster import (
    TrainingConfig,
    TrafficForecaster,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/synthetic/telemetry_forecasting.csv"


def _scaled_persistence(sequences: SequenceSet) -> np.ndarray:
    return sequences.scaled_target[sequences.origin_positions, None].repeat(
        len(sequences.horizon_steps), axis=1
    )


def _inverse_nonnegative(values: np.ndarray, scaler: Any) -> np.ndarray:
    return np.maximum(
        0.0,
        scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape),
    )


def _metric_rows(
    model_name: str,
    sequences: SequenceSet,
    scaled_predictions: np.ndarray,
    scaler: Any,
    *,
    target_column: str,
    predictions_are_raw: bool = False,
) -> list[dict[str, float | int | str | None]]:
    actual = sequences.batch_actual_targets(np.arange(len(sequences)))
    predicted = (
        np.asarray(scaled_predictions)
        if predictions_are_raw
        else _inverse_nonnegative(scaled_predictions, scaler)
    )
    if target_column in ("cpu_usage_percent", "memory_usage_percent"):
        predicted = np.clip(predicted, 0.0, 100.0)
    rows: list[dict[str, float | int | str | None]] = []
    for horizon_index, step in enumerate(sequences.horizon_steps):
        y_true = actual[:, horizon_index] if target_column == TARGET_COLUMN else None
        y_pred = predicted[:, horizon_index]
        if y_true is None:
            y_true = sequences.batch_actual_resources(
                np.arange(len(sequences))
            )[:, horizon_index, 0 if target_column == "cpu_usage_percent" else 1]
        rows.append(
            {
                "model": model_name,
                "target": target_column,
                "horizon_minutes": int(step * 5),
                "samples": int(len(y_true)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": (
                    float(r2_score(y_true, y_pred))
                    if len(y_true) >= 2 and np.ptp(y_true) > 0
                    else None
                ),
            }
        )
    return rows


def _classification_metrics(
    *,
    model_name: str,
    target_name: str,
    horizon_steps: tuple[int, ...],
    actual_positive: np.ndarray,
    predicted_positive: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Report positive-class precision/recall/F1 and auditable counts."""
    rows: list[dict[str, float | int | str]] = []
    for horizon_index, step in enumerate(horizon_steps):
        actual = actual_positive[:, horizon_index].astype(bool)
        predicted = predicted_positive[:, horizon_index].astype(bool)
        true_positive = int(np.count_nonzero(actual & predicted))
        false_positive = int(np.count_nonzero(~actual & predicted))
        false_negative = int(np.count_nonzero(actual & ~predicted))
        true_negative = int(np.count_nonzero(~actual & ~predicted))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "metric_type": "classification",
                "model": model_name,
                "target": target_name,
                "horizon_minutes": int(step * 5),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(actual.sum()),
                "predicted_positive_count": int(predicted.sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "samples": int(len(actual)),
            }
        )
    return rows


def _per_machine_metrics(
    sequences: SequenceSet,
    baseline: np.ndarray,
    lstm: np.ndarray,
    scaler: Any,
) -> list[dict[str, float | int | str]]:
    actual = sequences.batch_actual_targets(np.arange(len(sequences)))
    baseline_raw = _inverse_nonnegative(baseline, scaler)
    lstm_raw = _inverse_nonnegative(lstm, scaler)
    rows: list[dict[str, float | int | str]] = []
    machine_ids = sequences.machine_ids[sequences.origin_positions]
    for machine_id in np.unique(machine_ids):
        selected = machine_ids == machine_id
        for model_name, prediction in (
            ("persistence", baseline_raw),
            ("lstm", lstm_raw),
        ):
            for horizon_index, step in enumerate(sequences.horizon_steps):
                truth = actual[selected, horizon_index]
                estimate = prediction[selected, horizon_index]
                rows.append(
                    {
                        "machine_id": str(machine_id),
                        "model": model_name,
                        "horizon_minutes": int(step * 5),
                        "samples": int(len(truth)),
                        "mae": float(mean_absolute_error(truth, estimate)),
                        "rmse": float(np.sqrt(mean_squared_error(truth, estimate))),
                    }
                )
    return rows


def _horizon_summary(
    metric_rows: list[dict[str, float | int | str | None]],
    classification_rows: list[dict[str, float | int | str]],
) -> dict[str, dict[str, Any]]:
    """Summarize the 30/60/90/120-minute forecast windows in a machine-readable way."""
    summary: dict[str, dict[str, Any]] = {}
    for minutes in HORIZON_MINUTES:
        minute_key = f"{minutes}_minutes"
        summary[minute_key] = {
            "traffic_regression": {
                row["model"]: row
                for row in metric_rows
                if row.get("target") == TARGET_COLUMN and row.get("horizon_minutes") == minutes
            },
            "classification": {
                row["target"] + "_" + str(row["model"]): row
                for row in classification_rows
                if row.get("horizon_minutes") == minutes
            },
        }
    return summary


def _quality_notes(
    classification_rows: list[dict[str, float | int | str]],
) -> list[str]:
    """Keep poor burst recall visible without masking the underlying metric values."""
    notes: list[str] = []
    burst_rows = [
        row
        for row in classification_rows
        if row.get("target") == "traffic_burst" and row.get("model") == "lstm"
    ]
    if burst_rows:
        weakest = min(burst_rows, key=lambda row: float(row["recall"]))
        if float(weakest["recall"]) < 0.15:
            notes.append(
                "LSTM traffic burst recall remains low at "
                f"{weakest['horizon_minutes']} minutes ({weakest['recall']:.4f}); "
                "this poor performance is reported as-is and not hidden."
            )
    return notes


def _save_plots(
    test: SequenceSet,
    lstm_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    target_scaler: Any,
    output_dir: Path,
) -> None:
    actual = test.batch_actual_targets(np.arange(len(test)))
    model_values = _inverse_nonnegative(lstm_predictions, target_scaler)
    baseline_values = _inverse_nonnegative(baseline_predictions, target_scaler)
    machines = test.machine_ids[test.origin_positions]
    example_machine = str(np.unique(machines)[0])
    selected = np.flatnonzero(machines == example_machine)[:250]

    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(actual[selected, 1], label="Actual (60 min)", linewidth=1)
    axis.plot(model_values[selected, 1], label="LSTM (60 min)", linewidth=1)
    axis.plot(baseline_values[selected, 1], label="Persistence (60 min)", linewidth=1)
    axis.set(title=f"Actual vs predicted traffic — {example_machine}", ylabel="Bytes / 5 min")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=140)
    plt.close(fig)

    errors = model_values[:, 1] - actual[:, 1]
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.hist(errors, bins=50)
    axis.set(title="60-minute LSTM prediction error", xlabel="Predicted - actual bytes")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_error.png", dpi=140)
    plt.close(fig)

    horizon_mae = np.mean(np.abs(model_values - actual), axis=0)
    baseline_mae = np.mean(np.abs(baseline_values - actual), axis=0)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(HORIZON_MINUTES, horizon_mae, marker="o", label="LSTM")
    axis.plot(HORIZON_MINUTES, baseline_mae, marker="o", label="Persistence")
    axis.set(title="MAE by forecast horizon", xlabel="Horizon (minutes)", ylabel="MAE (bytes)")
    axis.set_xticks(HORIZON_MINUTES)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_performance.png", dpi=140)
    plt.close(fig)

    burst_sample = int(np.argmax(actual[:, 3]))
    burst_window = np.arange(max(0, burst_sample - 20), min(len(actual), burst_sample + 21))
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(burst_window, actual[burst_window, 3], label="Actual (120 min)", marker=".")
    axis.plot(burst_window, model_values[burst_window, 3], label="LSTM (120 min)", marker=".")
    axis.set(title="Highest observed test traffic events at 120-minute horizon", ylabel="Bytes / 5 min")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "burst_forecast_example.png", dpi=140)
    plt.close(fig)


def run_pipeline(
    data_path: str | Path = DEFAULT_DATA,
    *,
    output_dir: str | Path = PROJECT_ROOT / "outputs",
    model_dir: str | Path = PROJECT_ROOT / "models",
    epochs: int = 12,
    batch_size: int = 256,
    seed: int = 1780,
    scaler: str = "robust",
    lookback: int = 24,
    device: str = "auto",
    elevated_percentile: float = 0.85,
    burst_percentile: float = 0.95,
    pressure_percentile: float = 0.95,
) -> dict[str, Any]:
    """Run chronological fitting and save model, scalers, metrics, and forecasts."""
    output_path = Path(output_dir)
    models_path = Path(model_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    telemetry = load_telemetry(data_path, strict_frequency=True)
    featured = engineer_features(telemetry)
    boundaries = chronological_boundaries(featured)
    training_mask = utc_nanoseconds(featured["timestamp"]) <= boundaries.train_end_ns
    training_rows = featured.loc[training_mask].copy()
    preprocessor = TelemetryPreprocessor(scaler=scaler)  # type: ignore[arg-type]
    preprocessor.fit(training_rows)
    scaled_features, scaled_target = preprocessor.transform(featured)
    scaled_resource_targets = preprocessor.transform_resource_targets(featured)
    sequences = build_sequences(
        featured,
        scaled_features,
        scaled_target,
        lookback=lookback,
        scaled_resource_targets=scaled_resource_targets,
    )
    partitions = chronological_partitions(sequences, boundaries)
    if len(partitions["train"]) < 1 or len(partitions["validation"]) < 1:
        raise ValueError("The chronological split produced an empty training/validation set")

    thresholds = fit_traffic_thresholds(
        training_rows,
        elevated_percentile=elevated_percentile,
        burst_percentile=burst_percentile,
        pressure_percentile=pressure_percentile,
    )
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    forecaster = TrafficForecaster(
        input_size=len(FEATURE_COLUMNS),
        horizon_steps=sequences.horizon_steps,
        config=config,
    )
    training_result = forecaster.fit(partitions["train"], partitions["validation"])
    lstm_predictions = forecaster.predict_scaled(partitions["test"])
    baseline_predictions = _scaled_persistence(partitions["test"])

    metric_rows = _metric_rows(
        "persistence",
        partitions["test"],
        baseline_predictions,
        preprocessor.target_scaler,
        target_column=TARGET_COLUMN,
    )
    metric_rows.extend(
        _metric_rows(
            "lstm",
            partitions["test"],
            lstm_predictions[:, :, 0],
            preprocessor.target_scaler,
            target_column=TARGET_COLUMN,
        )
    )
    test = partitions["test"]
    test_indices = np.arange(len(test))
    actual_traffic = test.batch_actual_targets(test_indices)
    actual_resources = test.batch_actual_resources(test_indices)
    lstm_resource_predictions = np.stack(
        [
            preprocessor.inverse_resource_target(
                column, lstm_predictions[:, :, index]
            )
            for index, column in enumerate(
                ("cpu_usage_percent", "memory_usage_percent"), start=1
            )
        ],
        axis=2,
    ).clip(0.0, 100.0)
    origin_resource_actual = test.raw_resource_targets[test.origin_positions]
    persistence_resource_predictions = np.repeat(
        origin_resource_actual[:, None, :], len(test.horizon_steps), axis=1
    )
    for resource_index, column in enumerate(
        ("cpu_usage_percent", "memory_usage_percent")
    ):
        metric_rows.extend(
            _metric_rows(
                "persistence",
                test,
                persistence_resource_predictions[:, :, resource_index],
                preprocessor.resource_scalers[column],
                target_column=column,
                predictions_are_raw=True,
            )
        )
        metric_rows.extend(
            _metric_rows(
                "lstm",
                test,
                lstm_predictions[:, :, resource_index + 1],
                preprocessor.resource_scalers[column],
                target_column=column,
            )
        )

    actual_bursts = actual_traffic >= thresholds.burst_traffic
    persistence_bursts = baseline_predictions >= (
        preprocessor.target_scaler.transform(
            pd.DataFrame({TARGET_COLUMN: [thresholds.burst_traffic]})
        )[0, 0]
    )
    lstm_bursts = _inverse_nonnegative(
        lstm_predictions[:, :, 0], preprocessor.target_scaler
    ) >= thresholds.burst_traffic
    actual_pressure = (
        (actual_resources[:, :, 0] >= thresholds.cpu_pressure)
        | (actual_resources[:, :, 1] >= thresholds.memory_pressure)
    )
    persistence_pressure = (
        (persistence_resource_predictions[:, :, 0] >= thresholds.cpu_pressure)
        | (
            persistence_resource_predictions[:, :, 1]
            >= thresholds.memory_pressure
        )
    )
    predicted_pressure = (
        (lstm_resource_predictions[:, :, 0] >= thresholds.cpu_pressure)
        | (lstm_resource_predictions[:, :, 1] >= thresholds.memory_pressure)
    )
    classification_rows = _classification_metrics(
        model_name="persistence",
        target_name="traffic_burst",
        horizon_steps=test.horizon_steps,
        actual_positive=actual_bursts,
        predicted_positive=persistence_bursts,
    )
    classification_rows.extend(
        _classification_metrics(
            model_name="lstm",
            target_name="traffic_burst",
            horizon_steps=test.horizon_steps,
            actual_positive=actual_bursts,
            predicted_positive=lstm_bursts,
        )
    )
    classification_rows.extend(
        _classification_metrics(
            model_name="persistence",
            target_name="future_resource_pressure",
            horizon_steps=test.horizon_steps,
            actual_positive=actual_pressure,
            predicted_positive=persistence_pressure,
        )
    )
    classification_rows.extend(
        _classification_metrics(
            model_name="lstm",
            target_name="future_resource_pressure",
            horizon_steps=test.horizon_steps,
            actual_positive=actual_pressure,
            predicted_positive=predicted_pressure,
        )
    )
    per_machine_rows = _per_machine_metrics(
        partitions["test"],
        baseline_predictions,
        lstm_predictions[:, :, 0],
        preprocessor.target_scaler,
    )
    test_predictions_raw = np.concatenate(
        [
            _inverse_nonnegative(
                lstm_predictions[:, :, 0], preprocessor.target_scaler
            )[:, :, None],
            lstm_resource_predictions,
        ],
        axis=2,
    )
    historical = build_historical_inference_dataset(
        partitions["test"],
        test_predictions_raw,
        thresholds,
        featured,
    )

    forecaster.save(models_path / "traffic_lstm.pt")
    joblib.dump(
        {
            "feature_scaler": preprocessor.feature_scaler,
            "target_scaler": preprocessor.target_scaler,
            "resource_scalers": preprocessor.resource_scalers,
            "feature_columns": preprocessor.feature_columns,
            "scaler_name": preprocessor.scaler_name,
            "fitted_training_rows": preprocessor.fitted_training_rows,
            "training_end_ns": preprocessor.training_end_ns,
            "lookback": lookback,
        },
        models_path / "preprocessing.joblib",
    )
    save_thresholds(thresholds, models_path / "traffic_thresholds.json")
    preprocessing_config = {
        "feature_columns": preprocessor.feature_columns,
        "target_column": TARGET_COLUMN,
        "resource_target_columns": ["cpu_usage_percent", "memory_usage_percent"],
        "scaler": scaler,
        "scaler_fit_rows": preprocessor.fitted_training_rows,
        "scaler_training_end_utc": pd.Timestamp(
            preprocessor.training_end_ns, unit="ns", tz="UTC"
        ).isoformat(),
        "lookback_steps": lookback,
        "horizon_steps": list(sequences.horizon_steps),
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
    }
    (output_path / "preprocessing_config.json").write_text(
        json.dumps(preprocessing_config, indent=2), encoding="utf-8"
    )
    save_historical_inference(historical, output_path / "historical_inference.csv")
    save_prediction_payload(historical, output_path / "latest_forecasts.json")
    pd.DataFrame(metric_rows).to_csv(output_path / "evaluation_metrics.csv", index=False)
    pd.DataFrame(classification_rows).to_csv(
        output_path / "classification_metrics.csv", index=False
    )
    pd.DataFrame(metric_rows).loc[
        lambda frame: frame["target"].ne(TARGET_COLUMN)
    ].to_csv(output_path / "resource_regression_metrics.csv", index=False)
    pd.DataFrame(per_machine_rows).to_csv(
        output_path / "per_machine_metrics.csv", index=False
    )
    _save_plots(
        partitions["test"],
        lstm_predictions[:, :, 0],
        baseline_predictions,
        preprocessor.target_scaler,
        output_path,
    )

    quality_notes = _quality_notes(classification_rows)
    results = {
        "data_path": str(Path(data_path)),
        "rows": len(telemetry),
        "machines": int(telemetry["machine_id"].nunique()),
        "timestamps": int(telemetry["timestamp"].nunique()),
        "sampling_frequency": "5min",
        "lookback_steps": lookback,
        "horizons_minutes": list(HORIZON_MINUTES),
        "split_boundaries_utc": {
            name: pd.Timestamp(value, unit="ns", tz="UTC").isoformat()
            for name, value in (
                ("train_end", boundaries.train_end_ns),
                ("validation_start", boundaries.validation_start_ns),
                ("validation_end", boundaries.validation_end_ns),
                ("test_start", boundaries.test_start_ns),
            )
        },
        "sequence_counts": {name: len(part) for name, part in partitions.items()},
        "test_origin_policy": "origin and every horizon target must be within the test interval",
        "model": {
            "architecture": (
                "one 32-unit LSTM encoder with dropout and 12 linear outputs "
                "(4 horizons x traffic/CPU/memory)"
            ),
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout_probability": config.dropout,
            "seed": config.seed,
            "parameter_count": sum(
                parameter.numel() for parameter in forecaster.model.parameters()
            ),
            "device": training_result.device,
            "epochs_trained": training_result.epochs_trained,
            "best_validation_loss_scaled_mse": training_result.best_validation_loss,
            "history": training_result.history,
        },
        "thresholds": {
            "elevated_traffic_bytes": thresholds.elevated_traffic,
            "burst_traffic_bytes": thresholds.burst_traffic,
            "cpu_pressure_percent": thresholds.cpu_pressure,
            "memory_pressure_percent": thresholds.memory_pressure,
        },
        "evaluation_protocol": {
            "split": "chronological 70/15/15 by global timestamp",
            "test_policy": (
                "forecast origin and all horizon targets must fall within test interval"
            ),
            "traffic_target": (
                "network_rx_bytes + network_tx_bytes, bytes per five-minute interval"
            ),
            "resource_targets": "CPU and memory utilization percent",
            "burst_definition": (
                f"future traffic >= training {thresholds.burst_percentile:.0%} "
                f"quantile ({thresholds.burst_traffic:.6f} bytes)"
            ),
            "resource_pressure_definition": (
                f"future CPU >= training {thresholds.pressure_percentile:.0%} "
                f"CPU quantile ({thresholds.cpu_pressure:.6f}%) OR future memory "
                f">= training {thresholds.pressure_percentile:.0%} memory quantile "
                f"({thresholds.memory_pressure:.6f}%)"
            ),
            "threshold_fit_partition": "training only",
            "scaler_fit_partition": "training only",
            "regression_metrics": ["MAE", "RMSE", "R2"],
            "classification_metrics": [
                "precision",
                "recall",
                "F1",
                "positive support",
                "confusion counts",
            ],
        },
        "horizon_summary": _horizon_summary(metric_rows, classification_rows),
        "quality_notes": quality_notes,
        "traffic_regression_metrics": [
            row for row in metric_rows if row["target"] == TARGET_COLUMN
        ],
        "resource_regression_metrics": [
            row for row in metric_rows if row["target"] != TARGET_COLUMN
        ],
        "traffic_burst_classification_metrics": [
            row for row in classification_rows if row["target"] == "traffic_burst"
        ],
        "future_resource_pressure_classification_metrics": [
            row
            for row in classification_rows
            if row["target"] == "future_resource_pressure"
        ],
        "test_metrics": [
            row for row in metric_rows if row["target"] == TARGET_COLUMN
        ],
        "artifacts": [
            str(models_path / "traffic_lstm.pt"),
            str(models_path / "preprocessing.joblib"),
            str(models_path / "traffic_thresholds.json"),
            str(output_path / "historical_inference.csv"),
            str(output_path / "latest_forecasts.json"),
            str(output_path / "evaluation_metrics.csv"),
            str(output_path / "classification_metrics.csv"),
            str(output_path / "resource_regression_metrics.csv"),
            str(output_path / "per_machine_metrics.csv"),
            str(output_path / "preprocessing_config.json"),
            str(output_path / "actual_vs_predicted.png"),
            str(output_path / "prediction_error.png"),
            str(output_path / "horizon_performance.png"),
            str(output_path / "burst_forecast_example.png"),
            str(output_path / "evaluation_results.json"),
        ],
    }
    (output_path / "evaluation_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1780)
    parser.add_argument("--scaler", choices=("robust", "minmax"), default="robust")
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--elevated-percentile", type=float, default=0.85)
    parser.add_argument("--burst-percentile", type=float, default=0.95)
    parser.add_argument("--pressure-percentile", type=float, default=0.95)
    args = parser.parse_args()
    results = run_pipeline(
        args.data,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        scaler=args.scaler,
        lookback=args.lookback,
        device=args.device,
        elevated_percentile=args.elevated_percentile,
        burst_percentile=args.burst_percentile,
        pressure_percentile=args.pressure_percentile,
    )
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
