"""SHA-256 hash-chained audit log.

Every registry decision, drift action, and promotion is appended as an
entry whose hash covers the previous entry's hash, so tampering with any
historical entry breaks the chain. Verification replays the chain.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


def _entry_hash(prev_hash: str, payload: dict) -> str:
    material = prev_hash + json.dumps(payload, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class AuditEntry:
    seq: int
    event: str
    payload: dict
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "event": self.event,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


@dataclass
class AuditLog:
    entries: list[AuditEntry] = field(default_factory=list)

    def append(self, event: str, payload: dict) -> AuditEntry:
        prev = self.entries[-1].entry_hash if self.entries else "GENESIS"
        entry = AuditEntry(
            seq=len(self.entries),
            event=event,
            payload=payload,
            prev_hash=prev,
            entry_hash=_entry_hash(prev, {"seq": len(self.entries), "event": event, **payload}),
        )
        self.entries.append(entry)
        return entry

    def verify(self) -> bool:
        prev = "GENESIS"
        for entry in self.entries:
            if entry.prev_hash != prev:
                return False
            expected = _entry_hash(
                prev, {"seq": entry.seq, "event": entry.event, **entry.payload}
            )
            if expected != entry.entry_hash:
                return False
            prev = entry.entry_hash
        return True

    def to_dict(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]
