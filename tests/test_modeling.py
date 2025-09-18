import unittest
from pathlib import Path

from lolalytics import (
    LinearRegressionGD,
    build_feature_matrix,
    compute_metrics,
    load_html_from_file,
    scrape_champion_synergy,
    train_synergy_model,
)


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sample_path = Path(__file__).resolve().parents[1] / "lolalytics" / "data" / "sample_synergy.html"
        cls.records = scrape_champion_synergy("ahri", html=load_html_from_file(sample_path))

    def test_feature_matrix_has_expected_shape(self) -> None:
        features, targets = build_feature_matrix(self.records)
        self.assertEqual(len(features), len(targets))
        self.assertGreater(len(features), 0)
        self.assertEqual(len(features[0]), 8)

    def test_linear_regression_training_reduces_error(self) -> None:
        features, targets = build_feature_matrix(self.records)
        model = LinearRegressionGD(learning_rate=0.01, epochs=6000)
        model.fit(features, targets)
        predictions = model.predict(features)
        metrics = compute_metrics(predictions, targets)
        self.assertLess(metrics["mse"], 5.0)
        self.assertGreater(metrics["r2"], 0.65)

    def test_train_synergy_model_helper_returns_metrics(self) -> None:
        model, metrics = train_synergy_model(self.records, learning_rate=0.01, epochs=6000)
        self.assertIsNotNone(model.weights)
        self.assertIn("mse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
