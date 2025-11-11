"""Helpers for scraping champion synergy data from lolalytics.com."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import request, parse

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_NUXT_RE = re.compile(r"window\.__NUXT__\s*=\s*({.*?})\s*;\s*</script>", re.DOTALL)


def build_synergy_url(
    champion_slug: str,
    lane: Optional[str] = "middle",
    queue: Optional[int] = 420,
    region: Optional[str] = "world",
    patch: Optional[str] = None,
) -> str:
    """Construct the lolalytics synergy URL for a given champion.

    Args:
        champion_slug: URL slug for the champion (e.g. ``"ahri"``).
        lane: Optional lane filter to include in the query string.
        queue: Queue identifier (420 is solo queue). ``None`` omits the parameter.
        region: Region string accepted by lolalytics (``"world"`` by default).
        patch: Patch identifier such as ``"13.22"``. When ``None`` the site default
            is used.

    Returns:
        A formatted URL string.
    """

    base = f"https://lolalytics.com/lol/{champion_slug}/synergy/"
    params: Dict[str, Any] = {}
    if lane:
        params["lane"] = lane
    if queue is not None:
        params["queue"] = queue
    if region:
        params["region"] = region
    if patch:
        params["patch"] = patch

    if params:
        return base + "?" + parse.urlencode(params)
    return base


def fetch_url(url: str, timeout: float = 10.0) -> str:
    """Fetch HTML content from a URL using :mod:`urllib`.

    The helper sets a desktop browser user-agent because lolalytics rejects
    requests with the default python identifier.
    """

    req = request.Request(url, headers=DEFAULT_HEADERS)
    with request.urlopen(req, timeout=timeout) as response:  # type: ignore[arg-type]
        content = response.read()
    return content.decode("utf-8", errors="replace")


def load_html_from_file(path: str | Path) -> str:
    """Load HTML data from a local file."""

    file_path = Path(path)
    return file_path.read_text(encoding="utf-8")


def extract_nuxt_payload(html: str) -> Dict[str, Any]:
    """Extract the JSON payload embedded in the ``window.__NUXT__`` script.

    Args:
        html: The HTML document containing the ``window.__NUXT__`` assignment.

    Raises:
        ValueError: If the script tag cannot be located or the payload is not valid
            JSON.
    """

    match = _NUXT_RE.search(html)
    if not match:
        raise ValueError("Unable to locate window.__NUXT__ payload in the document")
    json_blob = match.group(1)
    try:
        return json.loads(json_blob)
    except json.JSONDecodeError as exc:  # pragma: no cover - exercised via tests
        raise ValueError("Invalid JSON payload in window.__NUXT__ script") from exc


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            raise TypeError
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_synergy_records(
    payload: Dict[str, Any],
    include_categories: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert a ``window.__NUXT__`` payload into tabular synergy records.

    Args:
        payload: Parsed JSON structure from :func:`extract_nuxt_payload`.
        include_categories: Optional iterable restricting which synergy buckets are
            included (e.g. ``{"good", "common"}``). ``None`` keeps every bucket.

    Returns:
        A list of dictionaries ready for CSV export or model training.
    """

    allowed = set(include_categories) if include_categories is not None else None
    results: List[Dict[str, Any]] = []

    data_sections = payload.get("data", [])
    if not isinstance(data_sections, list):
        return results

    for section in data_sections:
        if not isinstance(section, dict):
            continue

        champion_info: Dict[str, Any] = {}
        if "champion" in section and isinstance(section["champion"], dict):
            champion_info = section["champion"]
        else:
            champion_info = section

        synergy_data = champion_info.get("synergy")
        if not isinstance(synergy_data, dict):
            continue

        champion_name = champion_info.get("name") or section.get("name") or ""
        lane = champion_info.get("lane") or section.get("lane") or ""

        for synergy_type, entries in synergy_data.items():
            if allowed is not None and synergy_type not in allowed:
                continue
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                record = {
                    "champion": champion_name,
                    "lane": lane,
                    "synergy_type": synergy_type,
                    "ally": entry.get("ally") or entry.get("name") or "",
                    "ally_role": entry.get("allyRole") or entry.get("role") or "",
                    "win_rate": _to_float(entry.get("winRate")),
                    "delta": _to_float(entry.get("delta")),
                    "delta_normalized": _to_float(
                        entry.get("deltaNormalized") or entry.get("normalizedDelta")
                    ),
                    "pick_rate": _to_float(entry.get("pickRate")),
                    "games": _to_int(entry.get("games")),
                }
                results.append(record)

    return results


def save_records_to_csv(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """Persist parsed synergy records to a CSV file."""

    fieldnames = [
        "champion",
        "lane",
        "synergy_type",
        "ally",
        "ally_role",
        "win_rate",
        "delta",
        "delta_normalized",
        "pick_rate",
        "games",
    ]

    file_path = Path(path)
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def scrape_champion_synergy(
    champion_slug: str,
    lane: Optional[str] = "middle",
    queue: Optional[int] = 420,
    region: Optional[str] = "world",
    patch: Optional[str] = None,
    html: Optional[str] = None,
    fetch_html: Optional[Any] = None,
    include_categories: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """High level helper combining URL construction, fetching and parsing."""

    if html is None:
        url = build_synergy_url(champion_slug, lane=lane, queue=queue, region=region, patch=patch)
        fetcher = fetch_html or fetch_url
        html = fetcher(url)
    payload = extract_nuxt_payload(html)
    return parse_synergy_records(payload, include_categories=include_categories)


if __name__ == "__main__":  # pragma: no cover - manual usage helper
    sample_path = Path(__file__).resolve().parent / "data" / "sample_synergy.html"
    records = scrape_champion_synergy("ahri", html=load_html_from_file(sample_path))
    output_path = Path.cwd() / "ahri_synergy.csv"
    save_records_to_csv(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")
