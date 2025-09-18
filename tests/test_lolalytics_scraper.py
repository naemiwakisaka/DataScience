from pathlib import Path
import sys
from textwrap import dedent

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lolalytics_scraper import (
    LolAlyticsError,
    build_url,
    extract_nuxt_state,
    extract_synergies,
    find_candidate_tables,
    select_best_table,
)


HTML_WITH_WINDOW_STATE = dedent(
    """
    <html>
      <head>
        <script>
          window.__NUXT__ = {
            "data": [
              {
                "page": {
                  "champion": {
                    "synergy": {
                      "with": {
                        "rows": [
                          {
                            "champion": {"cid": 40, "name": "Janna"},
                            "wr": 54.32,
                            "matches": 1234,
                            "pr": 0.12
                          },
                          {
                            "champion": {"cid": 887, "name": "Aurora"},
                            "winRate": 0.5123,
                            "games": 532
                          }
                        ]
                      }
                    }
                  }
                }
              }
            ],
            "meta": {
              "champions": [
                {"cid": 40, "name": "Janna"},
                {"cid": 887, "name": "Aurora"},
                {"cid": 77, "alias": "Udyr"}
              ]
            }
          };
        </script>
      </head>
    </html>
    """
)


HTML_WITH_SCRIPT_TAG = dedent(
    """
    <html>
      <head>
        <script id="__NUXT_DATA__" type="application/json">
          {
            "data": [
              {
                "champion": {
                  "synergy": {
                    "with": [
                      {
                        "champion": {"cid": 31, "name": "Cho'Gath"},
                        "wr": 0.49,
                        "matches": "420"
                      }
                    ]
                  }
                }
              }
            ],
            "champions": [
              {"cid": 31, "name": "Cho'Gath"}
            ]
          }
        </script>
      </head>
    </html>
    """
)


def test_extract_nuxt_state_finds_window_payload():
    data = extract_nuxt_state(HTML_WITH_WINDOW_STATE)
    assert "data" in data
    assert data["meta"]["champions"][0]["name"] == "Janna"


def test_extract_nuxt_state_finds_script_tag_payload():
    data = extract_nuxt_state(HTML_WITH_SCRIPT_TAG)
    assert data["data"][0]["champion"]["synergy"]["with"][0]["champion"]["name"] == "Cho'Gath"


def test_extract_synergies_parses_records():
    records = extract_synergies(HTML_WITH_WINDOW_STATE)
    assert [record.champion for record in records] == ["Janna", "Aurora"]
    assert records[0].win_rate == pytest.approx(54.32)
    assert records[0].games == 1234
    assert records[0].pick_rate == pytest.approx(12.0)
    assert records[1].win_rate == pytest.approx(51.23)


def test_find_candidate_tables_identifies_synergy_section():
    nuxt = extract_nuxt_state(HTML_WITH_WINDOW_STATE)
    candidates = find_candidate_tables(nuxt)
    assert any("synergy" in path for path, _ in candidates)
    best = select_best_table(candidates)
    assert best is not None
    assert len(best) == 2


def test_extract_synergies_errors_when_missing_table():
    html = "<html><body>no nuxt data here</body></html>"
    with pytest.raises(LolAlyticsError):
        extract_synergies(html)


def test_build_url_handles_role_slugs():
    assert build_url("Trundle") == "https://lolalytics.com/lol/trundle/build/"
    assert build_url("Lee Sin", role="jungle") == "https://lolalytics.com/lol/leesin/jungle/build/"
