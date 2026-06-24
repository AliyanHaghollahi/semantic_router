"""
config.py — Loads config.yaml (or config.example.yaml fallback).
Provides a singleton `cfg` object used across all modules.
"""
import os
import yaml
from pathlib import Path

_ROOT = Path(__file__).parent

def _load_config() -> dict:
    cfg_path = _ROOT / "config.yaml"
    fallback  = _ROOT / "config.example.yaml"
    path = cfg_path if cfg_path.exists() else fallback
    with open(path) as f:
        return yaml.safe_load(f)

cfg = _load_config()
