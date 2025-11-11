# DataScience

Utilities for scraping and modeling League of Legends champion synergy data, inspired by the Lolalytics statistics shown in the screenshot.

## Project structure

- `lolalytics/scraping.py` – parse the `window.__NUXT__` payload from a Lolalytics synergy page into tabular records.
- `lolalytics/modeling.py` – create numerical features and train a lightweight gradient-descent regression model without external dependencies.
- `lolalytics/data/sample_synergy.html` – a fixture mirroring the Lolalytics synergy table for Ahri. Use it for offline experiments when the real website is not reachable.
- `tests/` – unit tests covering the scraping logic, CSV export and the regression workflow.

## Usage

1. **Scrape data**

   ```bash
   python -m lolalytics.scraping  # writes `ahri_synergy.csv` to the current directory using the sample HTML
   ```

   To scrape live data when network access is available:

   ```python
   from lolalytics.scraping import scrape_champion_synergy

   records = scrape_champion_synergy("ahri", lane="middle", queue=420, region="world")
   ```

2. **Train the model**

   ```bash
   python -m lolalytics.modeling
   ```

   or programmatically:

   ```python
   from lolalytics import load_html_from_file, scrape_champion_synergy, train_synergy_model
   from pathlib import Path

   sample_html = Path("lolalytics/data/sample_synergy.html")
   records = scrape_champion_synergy("ahri", html=load_html_from_file(sample_html))
   model, metrics = train_synergy_model(records)
   ```

## Tests

Run the unit test suite with:

```bash
python -m unittest discover -s tests
```
