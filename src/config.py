"""Configuration loading.

A config is just nested dictionaries read from YAML. Anything can be
overridden on the command line with `--set a.b.c=value`, which keeps
experiment sweeps in shell scripts instead of in edited source files.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class Config(dict):
    """A dict that also supports attribute-style and dotted access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Dict[str, Any] = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(dict(self))


def _coerce(raw: str) -> Any:
    """Turn a command-line string into the most specific type it looks like."""
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            pass
    if raw.startswith(("[", "{")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def load_config(path: str | Path = DEFAULT_CONFIG_PATH,
                overrides: Iterable[str] = ()) -> Config:
    """Read a YAML config and apply `key.subkey=value` overrides in order."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = Config(payload)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like a.b=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        config.set_path(dotted.strip(), _coerce(raw))
    return config


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the two flags every entry point in this repo shares."""
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="path to a YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override a config key, repeatable")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.overrides)
