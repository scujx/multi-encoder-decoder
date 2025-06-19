#!/usr/bin/env python
"""args_parser.py

Utility for merging YAML config files (with optional *includes*) and
command-line overrides into a single ``argparse.Namespace``.

* Supports nested *includes* (processed depth-first)
* Later keys override former ones (CLI → root config → included config)
* Uses ``pathlib.Path`` for robust path handling.

Example
-------
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument("--configs", default="configs/exp.yaml")
>>> args = parse_args(parser)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

YamlDict = Dict[str, Dict[str, Any]]

# -----------------------------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------------------------

def _load_yaml(path: Path) -> YamlDict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_into_namespace(ns: argparse.Namespace, cfg: YamlDict) -> None:
    """Flatten a two-level YAML mapping into attributes of *ns*."""
    for section in cfg.values():
        for key, val in section.items():
            setattr(ns, key, val)


def _process_includes(base_cfg: YamlDict, root_dir: Path) -> YamlDict:
    """Recursively merge ``includes`` files (depth-first)."""
    merged: YamlDict = {}
    includes = base_cfg.pop("includes", []) or []
    for inc in includes:
        inc_path = (root_dir / inc).expanduser().resolve()
        inc_cfg = _process_includes(_load_yaml(inc_path), inc_path.parent)
        merged.update(inc_cfg)
    merged.update(base_cfg)  # root overrides included
    return merged

# -----------------------------------------------------------------------------
# PUBLIC FUNCTION
# -----------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:  # noqa: D401
    """Return an argument namespace with YAML + CLI merged."""
    args = parser.parse_args()

    cfg_path = Path(args.configs).expanduser().resolve()
    root_cfg = _load_yaml(cfg_path)
    full_cfg = _process_includes(root_cfg, cfg_path.parent)

    # YAML → Namespace (default values)
    _merge_into_namespace(args, full_cfg)

    return args