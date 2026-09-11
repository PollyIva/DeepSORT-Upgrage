"""Caching utilities."""

import hashlib
import json
from pathlib import Path


def make_key(*args):
    raw = "_".join(map(str, args))
    return hashlib.md5(raw.encode()).hexdigest()


def load_cache(cache_dir, key):
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cache(cache_dir, key, data):
    path = Path(cache_dir) / f"{key}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
