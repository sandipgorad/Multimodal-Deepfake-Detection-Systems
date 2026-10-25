"""Provenance tracking: tamper-evident, hash-chained ledger of verified files.

Each record stores the file SHA-256 and the hash of the previous record, so
editing or deleting an old record breaks the chain. Stored as JSON lines
(works offline). `MongoLedger` mirrors it into MongoDB when available.
"""
import hashlib
import json
import time
from pathlib import Path

from src.metadata.metadata_check import sha256_of


class ProvenanceLedger:
    def __init__(self, path="data/provenance_ledger.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _records(self):
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _hash_record(rec):
        body = {k: v for k, v in rec.items() if k != "record_hash"}
        return hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()

    def add(self, file_path, verdict=None, source="upload"):
        recs = self._records()
        rec = {
            "file": Path(file_path).name,
            "sha256": sha256_of(file_path),
            "verdict": verdict,
            "source": source,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prev_hash": recs[-1]["record_hash"] if recs else "0" * 64,
        }
        rec["record_hash"] = self._hash_record(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def lookup(self, file_path):
        """Have we seen this exact file before? Returns matching records."""
        digest = sha256_of(file_path)
        return [r for r in self._records() if r["sha256"] == digest]

    def verify_chain(self):
        """True if no record has been altered or removed."""
        prev = "0" * 64
        for rec in self._records():
            if rec["prev_hash"] != prev or rec["record_hash"] != self._hash_record(rec):
                return False
            prev = rec["record_hash"]
        return True


class MongoLedger:
    """Optional MongoDB mirror (pip install pymongo)."""

    def __init__(self, uri="mongodb://localhost:27017", db="deepfake"):
        from pymongo import MongoClient
        self.col = MongoClient(uri)[db]["provenance"]

    def add(self, record):
        self.col.insert_one(dict(record))
