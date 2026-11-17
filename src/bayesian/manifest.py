"""Reproducibility manifests.

A manifest pins everything needed to reproduce a trained model exactly:
the seed, the full settings (hashed), the data version, and a hash of the
training data itself. Two runs with the same manifest hash are the same
run; anything else is a different run and must be recorded as such.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .config import Settings


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def data_hash(X: list[list[float]], y: list[int]) -> str:
    material = json.dumps({"X": X, "y": y}, sort_keys=True)
    return _sha256_text(material)


def config_hash(settings: Settings) -> str:
    return _sha256_text(json.dumps(settings.as_dict(), sort_keys=True))


@dataclass
class Manifest:
    model_name: str
    version: str
    seed: int
    config_hash: str
    data_version: str
    data_hash: str
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "data_hash": self.data_hash,
            "settings": self.settings,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def build_manifest(
    model_name: str,
    version: str,
    settings: Settings,
    X: list[list[float]],
    y: list[int],
) -> Manifest:
    return Manifest(
        model_name=model_name,
        version=version,
        seed=settings.seed,
        config_hash=config_hash(settings),
        data_version=settings.data_version,
        data_hash=data_hash(X, y),
        settings=settings.as_dict(),
    )
