#!/usr/bin/env python
"""data_processor.py

Lightweight loader / splitter for mixed-frequency Excel sheets.

* Sheet ``args.a1_sheet_name`` → auxiliary frequency A
* Sheet ``args.t_sheet_name``  → target   frequency T

Only minimal cleaning is applied; hook methods ``clean_data`` and
``feature_engineering`` can be overridden in subclasses.

Author : Jiaxi Liu
License: MIT
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DataPair = Tuple[pd.DataFrame, pd.DataFrame]

# -----------------------------------------------------------------------------

class DataProcessor:  # pylint: disable=too-few-public-methods
    """Read Excel, (optionally) clean / engineer features, split into train/test."""

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(self, args) -> None:  # args from YAML/CLI
        self.args = args
        self.file_path = Path("data") / f"{args.file_name}.xlsx"
        self.test_size = int(args.test_size) if args.test_size >= 1 else args.test_size

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def preprocess(self) -> Tuple[DataPair, DataPair]:
        """Full pipeline → (train_a, train_t), (test_a, test_t)."""
        data_a, data_t = self._read_excel()
        data_a, data_t = self.clean_data(data_a), self.clean_data(data_t)
        data_a, data_t = self.feature_engineering(data_a), self.feature_engineering(data_t)
        return self._split_data(data_a, data_t)

    # ------------------------------------------------------------------
    # steps
    # ------------------------------------------------------------------
    def _read_excel(self) -> DataPair:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Loading Excel → {self.file_path}")
        data_a = pd.read_excel(
            self.file_path,
            sheet_name=self.args.a1_sheet_name,
            index_col=0,
            dtype={0: str}
        )
        data_a.index = pd.to_datetime(data_a.index, format=self.args.date_format,
                                      errors='coerce')
        data_t = pd.read_excel(
            self.file_path,
            sheet_name=self.args.t_sheet_name,
            index_col=0,
            dtype={0: str}
        )
        data_t.index = pd.to_datetime(data_t.index, format=self.args.date_format,
                                      errors='coerce')
        return data_a, data_t

    # -------------------- cleaning hooks --------------------------------- #
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        """Override for specific cleaning (missing value handling, etc.)."""
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional feature crafting stub."""
        return df

    # -------------------- train / test split ----------------------------- #
    def _split_data(self, data_a: pd.DataFrame, data_t: pd.DataFrame) -> Tuple[DataPair, DataPair]:
        train_a, test_a = train_test_split(data_a, test_size=self.test_size, shuffle=False)
        train_t, test_t = train_test_split(data_t, test_size=self.test_size, shuffle=False)
        print(
            "Finished preprocessing → "
            f"train={len(train_t)} rows | test={len(test_t)} rows (target freq)"
        )
        return (train_a, train_t), (test_a, test_t)

    # ------------------------------------------------------------------
    #  date parser util
    # ------------------------------------------------------------------
    def _custom_date_parser(self, date_str: str | np.ndarray):  # type: ignore[override]
        """Handle non-existent 29/30 Feb by back-shifting to 28 Feb."""
        def fix(s: str) -> datetime:
            try:
                return datetime.strptime(s, self.args.date_format)
            except ValueError:
                if "2/29" in s:
                    return datetime.strptime(s.replace("2/29", "2/28") + " 06:00:00", "%Y/%m/%d %H:%M:%S")
                if "2/30" in s:
                    return datetime.strptime(s.replace("2/30", "2/28") + " 12:00:00", "%Y/%m/%d %H:%M:%S")
                raise

        if isinstance(date_str, np.ndarray):
            return [fix(str(dt)) for dt in date_str]
        if isinstance(date_str, str):
            return fix(date_str)
        return date_str  # already datetime
