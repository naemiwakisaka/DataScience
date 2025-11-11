import tempfile
import unittest
from pathlib import Path

from lolalytics import (
    build_synergy_url,
    extract_nuxt_payload,
    load_html_from_file,
    parse_synergy_records,
    save_records_to_csv,
    scrape_champion_synergy,
)


class ScrapingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_path = Path(__file__).resolve().parents[1] / "lolalytics" / "data" / "sample_synergy.html"
        cls.sample_html = load_html_from_file(cls.sample_path)

    def test_extract_nuxt_payload(self) -> None:
        payload = extract_nuxt_payload(self.sample_html)
        self.assertIn("data", payload)

    def test_parse_synergy_records(self) -> None:
        payload = extract_nuxt_payload(self.sample_html)
        records = parse_synergy_records(payload)
        self.assertEqual(len(records), 12)
        good_records = [row for row in records if row["synergy_type"] == "good"]
        self.assertEqual(len(good_records), 4)
        galio = next(row for row in records if row["ally"] == "Galio")
        self.assertAlmostEqual(galio["win_rate"], 54.82, places=2)

    def test_build_synergy_url(self) -> None:
        url = build_synergy_url("ahri", lane="middle", queue=420, region="world", patch="13.22")
        self.assertEqual(
            url,
            "https://lolalytics.com/lol/ahri/synergy/?lane=middle&queue=420&region=world&patch=13.22",
        )

    def test_scrape_champion_synergy_uses_provided_html(self) -> None:
        records = scrape_champion_synergy("ahri", html=self.sample_html)
        self.assertEqual(len(records), 12)

    def test_save_records_to_csv(self) -> None:
        payload = extract_nuxt_payload(self.sample_html)
        records = parse_synergy_records(payload)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "synergy.csv"
            save_records_to_csv(records, output_path)
            csv_contents = output_path.read_text(encoding="utf-8")
        self.assertIn("ally", csv_contents.splitlines()[0])
        self.assertIn("Galio", csv_contents)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
