from __future__ import annotations

import hashlib
import random


def _stable_u01(key: str, *, salt: str = "eval_sampling_v1") -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    # Use first 8 hex chars -> 32-bit int
    n = int(h[:8], 16)
    return n / 0xFFFFFFFF


def should_evaluate(*, query_id: str, sampling_rate: float) -> bool:
    """
    Deterministic sampling decision per query_id.

    - sampling_rate=0.0 -> never
    - sampling_rate=1.0 -> always
    """
    if sampling_rate <= 0.0:
        return False
    if sampling_rate >= 1.0:
        return True
    return _stable_u01(query_id) < sampling_rate

