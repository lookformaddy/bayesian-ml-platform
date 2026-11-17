"""Versioned model registry with promotion gates and rollback.

Versions move DRAFT → CANDIDATE → ACTIVE. Promotion is fail-closed: a
version must clear every gate — accuracy floor, AUC floor, calibration
(ECE) ceiling, and uncertainty-coverage floor — and carry a reproducibility
manifest. Rollback restores the previous ACTIVE version and is itself
audited.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .audit import AuditLog
from .config import Settings
from .evaluation import EvaluationReport
from .manifest import Manifest


@dataclass
class GateResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str = ""


@dataclass
class ModelVersion:
    name: str
    version: str
    stage: str = "DRAFT"  # DRAFT | CANDIDATE | ACTIVE | RETIRED
    metrics: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    gates: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage,
            "metrics": self.metrics,
            "manifest": self.manifest,
            "gates": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "observed": round(g.observed, 4),
                    "threshold": g.threshold,
                    "detail": g.detail,
                }
                for g in self.gates
            ],
        }


@dataclass
class ModelRegistry:
    settings: Settings
    versions: dict[str, ModelVersion] = field(default_factory=dict)
    audit: AuditLog = field(default_factory=AuditLog)
    active_version: str | None = None

    def _key(self, name: str, version: str) -> str:
        return f"{name}@{version}"

    def register(
        self,
        name: str,
        version: str,
        report: EvaluationReport,
        manifest: Manifest,
    ) -> ModelVersion:
        entry = ModelVersion(
            name=name,
            version=version,
            metrics=report.to_dict(),
            manifest=manifest.to_dict(),
        )
        self.versions[self._key(name, version)] = entry
        self.audit.append(
            "model_registered",
            {"model": name, "version": version, "metrics": entry.metrics},
        )
        return entry

    def evaluate_gates(self, version: ModelVersion) -> list[GateResult]:
        s = self.settings
        m = version.metrics
        gates = [
            GateResult(
                "accuracy_floor",
                m["accuracy"] >= s.accuracy_floor,
                m["accuracy"],
                s.accuracy_floor,
            ),
            GateResult(
                "auc_floor", m["auc"] >= s.auc_floor, m["auc"], s.auc_floor
            ),
            GateResult(
                "ece_ceiling", m["ece"] <= s.ece_ceiling, m["ece"], s.ece_ceiling
            ),
            GateResult(
                "coverage_floor",
                m["coverage_90"] >= s.coverage_floor,
                m["coverage_90"],
                s.coverage_floor,
            ),
            GateResult(
                "min_eval_cases",
                m["n"] >= s.min_eval_cases,
                float(m["n"]),
                float(s.min_eval_cases),
            ),
        ]
        version.gates = gates
        return gates

    def promote(self, name: str, version: str) -> tuple[bool, str]:
        key = self._key(name, version)
        entry = self.versions.get(key)
        if entry is None:
            return False, f"unknown version {key}"
        gates = self.evaluate_gates(entry)
        failed = [g for g in gates if not g.passed]
        if failed:
            self.audit.append(
                "promotion_rejected",
                {
                    "model": name,
                    "version": version,
                    "failed_gates": [g.name for g in failed],
                },
            )
            return False, "gates failed: " + ", ".join(g.name for g in failed)
        previous = self.active_version
        if previous is not None and previous in self.versions:
            self.versions[previous].stage = "RETIRED"
        entry.stage = "ACTIVE"
        self.active_version = key
        self.audit.append(
            "model_promoted",
            {"model": name, "version": version, "previous_active": previous},
        )
        return True, f"{key} promoted to ACTIVE"

    def rollback(self, name: str) -> tuple[bool, str]:
        """Restore the most recent RETIRED version of ``name`` to ACTIVE."""
        retired = [
            v
            for v in self.versions.values()
            if v.name == name and v.stage == "RETIRED"
        ]
        if not retired:
            return False, f"no previous version of {name} to roll back to"
        target = retired[-1]
        current = self.active_version
        if current is not None and current in self.versions:
            self.versions[current].stage = "RETIRED"
        target.stage = "ACTIVE"
        self.active_version = self._key(target.name, target.version)
        self.audit.append(
            "model_rolled_back",
            {
                "model": name,
                "version": target.version,
                "previous_active": current,
            },
        )
        return True, f"rolled back to {target.name}@{target.version}"

    def get_active(self) -> ModelVersion | None:
        if self.active_version is None:
            return None
        return self.versions.get(self.active_version)
