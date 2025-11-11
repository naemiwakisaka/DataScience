"""Utilities for scraping and modeling League of Legends synergy data."""

from .scraping import (
    build_synergy_url,
    extract_nuxt_payload,
    load_html_from_file,
    parse_synergy_records,
    save_records_to_csv,
    scrape_champion_synergy,
)
from .modeling import (
    build_feature_matrix,
    compute_metrics,
    hash_feature,
    LinearRegressionGD,
    train_synergy_model,
)

__all__ = [
    "build_synergy_url",
    "extract_nuxt_payload",
    "load_html_from_file",
    "parse_synergy_records",
    "save_records_to_csv",
    "scrape_champion_synergy",
    "build_feature_matrix",
    "compute_metrics",
    "hash_feature",
    "LinearRegressionGD",
    "train_synergy_model",
]
