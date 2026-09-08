"""Vault Hash ID assignment — REQ-ENGINE-3.11.

Format per design.md A.15: uppercase hex, `#` prefix (e.g. `#D3332A`).
Derived from the genome content plus a uniqueness salt so identical genomes
bred in different campaigns still get distinct IDs.
"""

from __future__ import annotations

import hashlib
import secrets

from app.genome.genome import Genome

HASH_ID_HEX_LENGTH = 6


def assign_hash_id(genome: Genome) -> str:
    salt = secrets.token_hex(8)
    digest = hashlib.sha256(f"{genome.to_dict()}{salt}".encode()).hexdigest()
    return f"#{digest[:HASH_ID_HEX_LENGTH].upper()}"
