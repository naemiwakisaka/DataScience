"""Lightweight machine learning helpers for synergy prediction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

SYNERGY_STRENGTH = {
    "legendary": 2.0,
    "strong": 1.5,
    "good": 1.0,
    "common": 0.0,
    "weak": -1.0,
    "poor": -1.5,
}


def hash_feature(value: str | None) -> float:
    """Map a string feature to a deterministic value in ``[-0.5, 0.5]``."""

    if not value:
        return 0.0
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / 2**64 - 0.5


def _synergy_strength_value(label: str | None) -> float:
    if not label:
        return 0.0
    return SYNERGY_STRENGTH.get(label.lower(), 0.0)


def build_feature_matrix(
    records: Sequence[Dict[str, object]]
) -> Tuple[List[List[float]], List[float]]:
    """Convert parsed synergy records into a design matrix and targets."""

    features: List[List[float]] = []
    targets: List[float] = []

    for record in records:
        row = [
            1.0,  # bias
            float(record.get("delta", 0.0)),
            float(record.get("delta_normalized", 0.0)),
            float(record.get("pick_rate", 0.0)),
            float(record.get("games", 0.0)) / 10000.0,
            _synergy_strength_value(record.get("synergy_type")),
            hash_feature(record.get("ally")),
            hash_feature(record.get("ally_role")),
        ]
        features.append(row)
        targets.append(float(record.get("win_rate", 0.0)))

    return features, targets


@dataclass
class LinearRegressionGD:
    """A minimal batch gradient-descent linear regressor."""

    learning_rate: float = 0.01
    epochs: int = 5000

    def __post_init__(self) -> None:
        self.weights: List[float] | None = None

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> "LinearRegressionGD":
        if not features:
            raise ValueError("No features supplied to fit the model")
        n_samples = len(features)
        n_features = len(features[0])
        weights = [0.0] * n_features

        for epoch in range(self.epochs):
            gradients = [0.0] * n_features
            for row, target in zip(features, targets):
                prediction = self._predict_row(weights, row)
                error = prediction - target
                for index, value in enumerate(row):
                    gradients[index] += error * value
            for index in range(n_features):
                weights[index] -= (self.learning_rate / n_samples) * gradients[index]
        self.weights = weights
        return self

    @staticmethod
    def _predict_row(weights: Sequence[float], row: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(weights, row))

    def predict(self, features: Sequence[Sequence[float]]) -> List[float]:
        if self.weights is None:
            raise ValueError("The model has not been fitted yet")
        return [self._predict_row(self.weights, row) for row in features]

    def predict_single(self, row: Sequence[float]) -> float:
        if self.weights is None:
            raise ValueError("The model has not been fitted yet")
        return self._predict_row(self.weights, row)


def compute_metrics(predictions: Sequence[float], targets: Sequence[float]) -> Dict[str, float]:
    """Calculate regression metrics (MSE, MAE, R^2)."""

    total = len(targets)
    if total == 0:
        raise ValueError("Cannot compute metrics without targets")

    mse = sum((pred - tgt) ** 2 for pred, tgt in zip(predictions, targets)) / total
    mae = sum(abs(pred - tgt) for pred, tgt in zip(predictions, targets)) / total

    mean_target = sum(targets) / total
    ss_total = sum((tgt - mean_target) ** 2 for tgt in targets)
    ss_res = sum((tgt - pred) ** 2 for pred, tgt in zip(predictions, targets))
    r2 = 1 - ss_res / ss_total if ss_total else 0.0

    return {"mse": mse, "mae": mae, "r2": r2}


def train_synergy_model(
    records: Sequence[Dict[str, object]],
    learning_rate: float = 0.01,
    epochs: int = 4000,
) -> Tuple[LinearRegressionGD, Dict[str, float]]:
    """Train a regression model on synergy records and return diagnostics."""

    features, targets = build_feature_matrix(records)
    model = LinearRegressionGD(learning_rate=learning_rate, epochs=epochs)
    model.fit(features, targets)
    predictions = model.predict(features)
    metrics = compute_metrics(predictions, targets)
    return model, metrics


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from pathlib import Path
    from .scraping import load_html_from_file, scrape_champion_synergy

    sample_html = Path(__file__).resolve().parent / "data" / "sample_synergy.html"
    data = scrape_champion_synergy("ahri", html=load_html_from_file(sample_html))
    model, metrics = train_synergy_model(data)
    print("Trained model weights:", model.weights)
    print("Metrics:", metrics)
