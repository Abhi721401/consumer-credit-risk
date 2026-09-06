"""
data_ingestion.py
------------------
Loads the raw "Default of Credit Card Clients" dataset.

Design notes:
- No hard-coded absolute paths; the raw file location is passed in or
  read from an environment variable / config default relative to the
  project root.
- Does not assume the schema — callers should run data_validation.py
  before trusting column names/types.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the raw credit-card-clients CSV into a DataFrame.

    Parameters
    ----------
    path : str | Path | None
        Path to the raw CSV. If None, uses the CREDIT_RISK_RAW_PATH
        environment variable if set, otherwise the default project
        path data/raw/UCI_Credit_Card.csv.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataframe exactly as read from disk.
    """
    resolved_path = Path(path) if path else Path(os.environ.get("CREDIT_RISK_RAW_PATH", DEFAULT_RAW_PATH))

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {resolved_path}. "
            "Download the 'Default of Credit Card Clients' dataset and place it there, "
            "or set CREDIT_RISK_RAW_PATH."
        )

    logger.info("Loading raw data from %s", resolved_path)
    df = pd.read_csv(resolved_path)
    logger.info("Loaded raw data: %d rows x %d columns", df.shape[0], df.shape[1])
    return df


if __name__ == "__main__":
    data = load_raw_data()
    print(data.shape)
    print(data.columns.tolist())
