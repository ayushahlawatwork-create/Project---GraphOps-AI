"""Direct multi-horizon LSTM forecaster for aggregate network traffic."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.sequence_builder import SequenceSet


@dataclass(frozen=True)
class TrainingConfig:
    """Small LSTM configuration suitable for a local university project."""

    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.10
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 12
    patience: int = 3
    seed: int = 1780
    device: str = "auto"


class LSTMNetwork(nn.Module):
    """One LSTM encoder with four direct output heads (one per horizon)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        horizon_count: int,
        target_count: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)
        self.horizon_count = horizon_count
        self.target_count = target_count

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(values)
        output = self.output(self.dropout(encoded[:, -1, :]))
        return output.view(-1, self.horizon_count, self.target_count)


@dataclass
class TrainingResult:
    history: list[dict[str, float]]
    best_validation_loss: float
    epochs_trained: int
    device: str


class TrafficForecaster:
    """Train, checkpoint, and batch-predict traffic at configured horizons."""

    def __init__(
        self,
        input_size: int,
        horizon_steps: tuple[int, ...] = (6, 12, 18, 24),
        config: TrainingConfig | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.input_size = input_size
        self.horizon_steps = horizon_steps
        self.target_count = 3
        self.set_seed(self.config.seed)
        self.device = self._select_device(self.config.device)
        self.model = LSTMNetwork(
            input_size=input_size,
            output_size=len(horizon_steps) * self.target_count,
            horizon_count=len(horizon_steps),
            target_count=self.target_count,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this PyTorch build")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _loss(self, sequences: SequenceSet, batch_size: int) -> float:
        self.model.eval()
        total_loss = 0.0
        sample_count = 0
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                indices = np.arange(start, min(start + batch_size, len(sequences)))
                x = torch.as_tensor(
                    sequences.batch_inputs(indices), device=self.device
                )
                y = torch.as_tensor(
                    sequences.batch_targets(indices), device=self.device
                )
                loss = nn.functional.mse_loss(self.model(x), y, reduction="sum")
                total_loss += float(loss.item())
                sample_count += len(indices)
        return total_loss / max(
            sample_count * len(self.horizon_steps) * self.target_count, 1
        )

    def fit(
        self, training: SequenceSet, validation: SequenceSet
    ) -> TrainingResult:
        """Train in chronological order with validation-based early stopping."""
        if not len(training) or not len(validation):
            raise ValueError("Training and validation sequences must both be non-empty")
        if training.feature_count != self.input_size:
            raise ValueError("Sequence feature count does not match model input_size")
        if training.scaled_resource_targets is None:
            raise ValueError("Training sequences must include scaled CPU and memory targets")
        if validation.scaled_resource_targets is None:
            raise ValueError("Validation sequences must include scaled CPU and memory targets")
        self.set_seed(self.config.seed)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            sample_count = 0
            for start in range(0, len(training), self.config.batch_size):
                indices = np.arange(
                    start, min(start + self.config.batch_size, len(training))
                )
                inputs = torch.as_tensor(
                    training.batch_inputs(indices), device=self.device
                )
                targets = torch.as_tensor(
                    training.batch_targets(indices), device=self.device
                )
                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(inputs)
                loss = nn.functional.mse_loss(predictions, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.detach().item()) * len(indices)
                sample_count += len(indices)

            training_loss = total_loss / max(sample_count, 1)
            validation_loss = self._loss(validation, self.config.batch_size)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if best_state is None:
            raise RuntimeError("Training did not produce a valid model checkpoint")
        self.model.load_state_dict(best_state)
        return TrainingResult(
            history=history,
            best_validation_loss=best_loss,
            epochs_trained=len(history),
            device=str(self.device),
        )

    def predict_scaled(self, sequences: SequenceSet) -> np.ndarray:
        """Predict each configured horizon without recursive feedback."""
        self.model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.config.batch_size):
                indices = np.arange(
                    start, min(start + self.config.batch_size, len(sequences))
                )
                inputs = torch.as_tensor(
                    sequences.batch_inputs(indices), device=self.device
                )
                predictions.append(
                    self.model(inputs).detach().cpu().numpy().astype(np.float32)
                )
        if not predictions:
            return np.empty((0, len(self.horizon_steps)), dtype=np.float32)
        return np.concatenate(predictions, axis=0)

    def save(self, path: str | Path) -> None:
        """Save model state and the architecture/config needed to reload it."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_size": self.input_size,
                "horizon_steps": self.horizon_steps,
                "target_count": self.target_count,
                "config": asdict(self.config),
                "model_state": self.model.state_dict(),
            },
            checkpoint_path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> "TrafficForecaster":
        """Restore a saved forecaster on an available or requested device."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config_values = dict(checkpoint["config"])
        config_values["device"] = device
        forecaster = cls(
            input_size=int(checkpoint["input_size"]),
            horizon_steps=tuple(checkpoint["horizon_steps"]),
            config=TrainingConfig(**config_values),
        )
        forecaster.target_count = int(checkpoint.get("target_count", 1))
        forecaster.model = LSTMNetwork(
            input_size=forecaster.input_size,
            output_size=len(forecaster.horizon_steps) * forecaster.target_count,
            horizon_count=len(forecaster.horizon_steps),
            target_count=forecaster.target_count,
            hidden_size=forecaster.config.hidden_size,
            num_layers=forecaster.config.num_layers,
            dropout=forecaster.config.dropout,
        ).to(forecaster.device)
        forecaster.model.load_state_dict(checkpoint["model_state"])
        return forecaster
