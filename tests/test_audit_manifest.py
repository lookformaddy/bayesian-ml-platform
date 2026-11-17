import pytest

from bayesian.audit import AuditLog
from bayesian.config import Settings
from bayesian.manifest import build_manifest, config_hash, data_hash


def test_audit_chain_verifies():
    log = AuditLog()
    log.append("model_registered", {"model": "m", "version": "v1"})
    log.append("model_promoted", {"model": "m", "version": "v1"})
    assert log.verify() is True
    assert len(log.entries) == 2
    assert log.entries[0].prev_hash == "GENESIS"


def test_audit_tamper_detected():
    log = AuditLog()
    log.append("a", {"x": 1})
    log.append("b", {"x": 2})
    log.entries[0].payload["x"] = 99
    assert log.verify() is False


def test_audit_empty_verifies():
    assert AuditLog().verify() is True


def test_manifest_is_deterministic():
    s = Settings()
    X = [[0.1, 0.2], [0.3, 0.4]]
    y = [0, 1]
    m1 = build_manifest("m", "v1", s, X, y)
    m2 = build_manifest("m", "v1", s, X, y)
    assert m1.to_dict() == m2.to_dict()
    assert len(m1.config_hash) == 64
    assert len(m1.data_hash) == 64


def test_data_hash_changes_with_data():
    assert data_hash([[0.1]], [0]) != data_hash([[0.2]], [0])
    assert config_hash(Settings(seed=1)) != config_hash(Settings(seed=2))
