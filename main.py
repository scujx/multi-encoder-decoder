#!/usr/bin/env python
"""main.py

Entry point for training / evaluating the dual-frequency MED model.

Usage
-----
$ python main.py --configs configs/0_simu/med.yaml \
                --add_info v1

Command-line flags supplement YAML config (see ``args_parser.parse_args``).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from args_parser import parse_args
from data_processor import DataProcessor
from trainer import ModelTrainer  # renamed module

# -----------------------------------------------------------------------------
# Argument stub (only override values not in YAML)
# -----------------------------------------------------------------------------
_cli_parser = argparse.ArgumentParser(description="Train MED on mixed-frequency data")
_cli_parser.add_argument("--configs", default="configs/0_simu/med.yaml")
_cli_parser.add_argument("--output_path", default="./output", type=Path)
_cli_parser.add_argument("--model_dir", default="")
_cli_parser.add_argument("--model_name", default="")
_cli_parser.add_argument("--verbose", default=10, type=int)
_cli_parser.add_argument("--add_info", default="", type=str)


def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1) load YAML + CLI overrides
    # ------------------------------------------------------------------
    args = parse_args(_cli_parser)

    # ------------------------------------------------------------------
    # 2) data preparation
    # ------------------------------------------------------------------
    train_data, test_data = DataProcessor(args).preprocess()

    # ------------------------------------------------------------------
    # 3) model training / testing
    # ------------------------------------------------------------------
    trainer = ModelTrainer(train_data, test_data, args)
    trainer.train()  # K-fold training
    print(f"Training finished in {time.time() - t0:.2f}s")

    trainer.test()   # evaluate on hold-out test set
    print(f"Testing finished in {time.time() - t0:.2f}s total")


if __name__ == "__main__":
    main()