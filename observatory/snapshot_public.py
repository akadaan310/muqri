#!/usr/bin/env python3
"""Capture a review snapshot of the canonical machine-review JSON.

The canonical representation is the LIVE endpoint GET /api/observatory/public, generated on each request
from the current data. observatory/data/public_review.json is only a point-in-time copy of it for
reviewers who cannot reach the endpoint; it is labelled as such inside the file and is never read by
the Observatory.

    .venv/bin/python observatory/snapshot_public.py        # needs the Observatory on 127.0.0.1:8095
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "observatory/data/public_review.json"
SRC = os.environ.get("OBSERVATORY_URL", "http://127.0.0.1:8095") + "/api/observatory/public"


def main() -> int:
    with urllib.request.urlopen(SRC, timeout=60) as r:
        body = json.loads(r.read())
    snap = {"review_snapshot": {
        "canonical_source": "GET /api/observatory/public (live, generated per request)",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "observatory_revision_at_capture": body.get("revisions", {}).get("observatory_running"),
        "note": "Point-in-time copy for review only. Not authoritative and not updated automatically; "
                "where it differs from the live endpoint, the live endpoint is correct."}, **body}
    OUT.write_text(json.dumps(snap, ensure_ascii=False, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)} ({snap['review_snapshot']['captured_at']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
