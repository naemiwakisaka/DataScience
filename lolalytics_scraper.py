"""Utilities for scraping synergy information from Lolalytics champion build pages.

This module focuses on extracting the ally synergy data that appears on champion build
pages such as ``https://lolalytics.com/lol/trundle/build/``. The real website is rendered
by a Nuxt application, which means the interesting data is embedded inside a JavaScript
object (``window.__NUXT__`` or a ``<script id="__NUXT_DATA__">`` element).  We treat the
HTML as an opaque string, pull that JSON blob out, and then search it for the structures
that look like "synergy" tables.

Because the execution environment used for the kata does not have outbound internet
access, the :func:`fetch_html` helper will raise a ``URLError`` when used directly.  The
logic is still implemented so that the script works on a local machine with internet
access, but for testing we rely on synthetic HTML snippets that mimic the structure of
Lolalytics responses.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen


LOLALYTICS_BASE_URL = "https://lolalytics.com"


class LolAlyticsError(RuntimeError):
    """Custom exception used for scraper specific errors."""


@dataclass
class SynergyRecord:
    """Represents a single ally synergy entry for a champion."""

    champion: str
    win_rate: Optional[float]
    games: Optional[int]
    pick_rate: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

    def formatted(self) -> str:
        """Return a human readable representation of the record."""

        win_rate_text = f"{self.win_rate:.2f}%" if self.win_rate is not None else "?"
        games_text = f"{self.games:,}" if self.games is not None else "?"
        pick_rate_text = (
            f" ({self.pick_rate:.2f}% pick)" if self.pick_rate is not None else ""
        )
        return f"{self.champion:<15} | Win Rate: {win_rate_text} | Games: {games_text}{pick_rate_text}"


def fetch_html(url: str, *, user_agent: str = "Mozilla/5.0", timeout: float = 10.0) -> str:
    """Fetch the raw HTML for ``url``.

    Parameters
    ----------
    url:
        The full URL that should be downloaded.
    user_agent:
        User agent sent in the ``Request`` headers.  Lolalytics blocks the default
        Python user agent, so the default mimics a standard browser.
    timeout:
        Timeout in seconds for the HTTP request.

    Returns
    -------
    str
        The HTML payload as a text string.
    """

    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:  # type: ignore[arg-type]
        return response.read().decode("utf-8", errors="replace")


SCRIPT_TAG_PATTERN = re.compile(
    r"<script[^>]+id=\"__NUXT_DATA__\"[^>]*>(?P<data>.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
WINDOW_STATE_PATTERN = re.compile(r"window\.__NUXT__\s*=\s*", re.IGNORECASE)


def _extract_balanced_json(text: str, start_index: int) -> Optional[str]:
    """Extract a JSON-ish object starting at ``start_index``.

    Lolalytics embeds a full JSON object immediately after ``window.__NUXT__ =``.
    The payload uses braces/brackets that need to be balanced.  This helper walks the
    string and returns the substring that spans the JSON structure.
    """

    # Move index to the first opening brace/bracket
    while start_index < len(text) and text[start_index].isspace():
        start_index += 1
    if start_index >= len(text) or text[start_index] not in "[{":
        return None

    stack = []
    index = start_index
    while index < len(text):
        char = text[index]
        if char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack:
                break
            opening = stack.pop()
            if (opening == "{" and char != "}") or (opening == "[" and char != "]"):
                return None
            if not stack:
                return text[start_index : index + 1]
        elif char in "\"'":
            # Skip over quoted strings to avoid mismatching braces inside strings.
            quote = char
            index += 1
            while index < len(text):
                current = text[index]
                if current == "\\":
                    index += 2
                    continue
                if current == quote:
                    break
                index += 1
        index += 1
    return None


def extract_nuxt_state(html: str) -> Dict[str, Any]:
    """Extract the ``window.__NUXT__`` object from the provided HTML.

    Returns the parsed JSON as a Python dictionary.  A :class:`LolAlyticsError` is
    raised if no Nuxt payload can be located.
    """

    script_match = SCRIPT_TAG_PATTERN.search(html)
    json_payload: Optional[str] = None
    if script_match:
        json_payload = script_match.group("data").strip()
    else:
        window_match = WINDOW_STATE_PATTERN.search(html)
        if window_match:
            start = window_match.end()
            candidate = _extract_balanced_json(html, start)
            if candidate is not None:
                json_payload = candidate

    if not json_payload:
        raise LolAlyticsError("Nuxt payload not found in the provided HTML.")

    cleaned = (
        json_payload.strip()
        .rstrip(";")
        .replace("undefined", "null")
        .replace("NaN", "null")
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
        raise LolAlyticsError("Failed to parse Nuxt payload.") from exc


def build_champion_index(data: Any) -> Dict[int, str]:
    """Create a mapping from champion ids to champion names.

    The Nuxt state includes multiple repeated structures with ``cid`` (champion id)
    alongside a human readable ``name``/``alias``.  We recursively traverse the
    payload to harvest those relationships.
    """

    mapping: Dict[int, str] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            cid: Optional[int] = None
            name: Optional[str] = None
            for key, value in node.items():
                lowered = key.lower()
                if lowered in {"cid", "championid", "champion_id"}:
                    try:
                        cid = int(value)
                    except (TypeError, ValueError):
                        cid = None
                elif lowered in {"name", "alias", "slug", "champion"} and isinstance(
                    value, str
                ):
                    name = value
            if cid is not None and name:
                mapping.setdefault(cid, name)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(data)
    return mapping


SynergyEntry = Dict[str, Any]
CandidateTable = Tuple[str, List[SynergyEntry]]


TABLE_LIST_KEYS = ("rows", "list", "items", "values", "data")


def _normalise_list(value: Any) -> Optional[List[SynergyEntry]]:
    """Return ``value`` as a list of dictionaries when possible."""

    if isinstance(value, list) and all(isinstance(v, dict) for v in value):
        return value  # type: ignore[return-value]
    if isinstance(value, dict):
        for key in TABLE_LIST_KEYS:
            nested = value.get(key)
            if isinstance(nested, list) and all(isinstance(v, dict) for v in nested):
                return nested  # type: ignore[return-value]
    return None


WIN_RATE_KEYS = ("wr", "winrate", "win_rate", "winratio", "winratio", "winpercent", "win_percent", "winRate", "winPercent")
GAME_COUNT_KEYS = ("matches", "games", "count", "sample", "n", "numGames", "play")
PICK_RATE_KEYS = ("pr", "pickrate", "pick_rate", "pickRate", "pickPercent")
PARTNER_KEYS = ("champion", "ally", "with", "partner", "unit")
NAME_KEYS = ("name", "alias", "slug", "id", "champion")


def _looks_like_synergy_entry(entry: SynergyEntry) -> bool:
    if not isinstance(entry, dict):
        return False
    has_wr = any(key in entry for key in WIN_RATE_KEYS)
    has_partner = any(key in entry for key in PARTNER_KEYS) or any(
        isinstance(entry.get(key), str) for key in NAME_KEYS
    )
    return has_wr and has_partner


def find_candidate_tables(data: Any) -> List[CandidateTable]:
    """Search the Nuxt state for lists that resemble ally synergy tables."""

    candidates: List[CandidateTable] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = f"{path}.{key}" if path else key
                normalised = _normalise_list(value)
                if normalised and normalised and all(
                    _looks_like_synergy_entry(entry) for entry in normalised
                ):
                    candidates.append((next_path, normalised))
                visit(value, next_path)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]")

    visit(data, "")
    return candidates


def _table_score(path: str, entries: Sequence[SynergyEntry]) -> Tuple[int, int, int]:
    lowered = path.lower()
    keyword_score = 0
    for keyword in ("synergy", "duo", "with"):
        if keyword in lowered:
            keyword_score += 1
    return (keyword_score, sum(1 for entry in entries if entry), len(entries))


def select_best_table(candidates: List[CandidateTable]) -> Optional[List[SynergyEntry]]:
    if not candidates:
        return None
    path, entries = max(candidates, key=lambda item: _table_score(item[0], item[1]))
    return list(entries)


def _first_numeric(entry: SynergyEntry, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in entry:
            value = entry[key]
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value.strip().rstrip("%"))
                except ValueError:
                    continue
    return None


def _champion_name_from_entry(entry: SynergyEntry, mapping: Dict[int, str]) -> Optional[str]:
    for key in PARTNER_KEYS:
        value = entry.get(key)
        if isinstance(value, dict):
            for name_key in NAME_KEYS:
                name = value.get(name_key)
                if isinstance(name, str):
                    return name
            cid = value.get("cid") or value.get("championId") or value.get("champion_id")
            if isinstance(cid, (int, str)):
                try:
                    cid_int = int(cid)
                except ValueError:
                    cid_int = None
                else:
                    if cid_int in mapping:
                        return mapping[cid_int]
        elif isinstance(value, str):
            return value
    for key in NAME_KEYS:
        name = entry.get(key)
        if isinstance(name, str):
            return name
    for key in ("cid", "championId", "champion_id"):
        cid = entry.get(key)
        if isinstance(cid, (int, str)):
            try:
                cid_int = int(cid)
            except ValueError:
                continue
            if cid_int in mapping:
                return mapping[cid_int]
    return None


def _normalise_win_rate(win_rate: Optional[float]) -> Optional[float]:
    if win_rate is None:
        return None
    if win_rate <= 1:
        win_rate *= 100
    return round(win_rate, 2)


def _normalise_pick_rate(pick_rate: Optional[float]) -> Optional[float]:
    if pick_rate is None:
        return None
    if pick_rate <= 1:
        pick_rate *= 100
    return round(pick_rate, 2)


def _normalise_games(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    return max(int(round(value)), 0)


def extract_synergies(html: str) -> List[SynergyRecord]:
    """High level helper that extracts ally synergy data from Lolalytics HTML."""

    nuxt_state = extract_nuxt_state(html)
    champion_index = build_champion_index(nuxt_state)
    candidates = find_candidate_tables(nuxt_state)
    table = select_best_table(candidates)
    if not table:
        raise LolAlyticsError("Unable to locate synergy table within Nuxt payload.")

    records: List[SynergyRecord] = []
    for entry in table:
        name = _champion_name_from_entry(entry, champion_index)
        if not name:
            continue
        win_rate = _normalise_win_rate(_first_numeric(entry, WIN_RATE_KEYS))
        pick_rate = _normalise_pick_rate(_first_numeric(entry, PICK_RATE_KEYS))
        games = _normalise_games(_first_numeric(entry, GAME_COUNT_KEYS))
        records.append(
            SynergyRecord(
                champion=name,
                win_rate=win_rate,
                games=games,
                pick_rate=pick_rate,
                raw=entry,
            )
        )

    records.sort(key=lambda record: (record.win_rate or 0, record.games or 0), reverse=True)
    return records


def build_url(champion: str, role: Optional[str] = None, region: str = "world") -> str:
    """Construct the Lolalytics URL for a champion build page."""

    slug = champion.lower().replace(" ", "")
    role_fragment = f"{role}/" if role else ""
    return f"{LOLALYTICS_BASE_URL}/lol/{slug}/{role_fragment}build/"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape ally synergies from Lolalytics")
    parser.add_argument("champion", help="Champion slug, e.g. trundle")
    parser.add_argument(
        "--role",
        help="Role/lane used on Lolalytics (top, jungle, support, etc.)",
        default=None,
    )
    parser.add_argument(
        "--html-file",
        help="Path to a local HTML file (useful when developing offline)",
        default=None,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top synergy entries to display",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the raw dictionary for each synergy entry",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    url = build_url(args.champion, args.role)

    try:
        if args.html_file:
            with open(args.html_file, "r", encoding="utf-8") as handle:
                html = handle.read()
        else:
            html = fetch_html(url)
    except FileNotFoundError as exc:
        print(f"Failed to read HTML file: {exc}", file=sys.stderr)
        return 2
    except URLError as exc:
        print(
            "Unable to download Lolalytics page. If you are working offline use --html-file.",
            file=sys.stderr,
        )
        print(f"URL error: {exc}", file=sys.stderr)
        return 2

    try:
        synergies = extract_synergies(html)
    except LolAlyticsError as exc:
        print(f"Error while parsing synergy data: {exc}", file=sys.stderr)
        return 1

    if not synergies:
        print("No synergy data found.")
        return 0

    limit = args.top if args.top > 0 else len(synergies)
    for record in synergies[:limit]:
        print(record.formatted())
        if args.show_raw and record.raw is not None:
            print(json.dumps(record.raw, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
