# main.py
# Build a single, ever-growing CSV from all structured JSONL files.
# Reads:  gs://<bucket>/<STRUCTURED_PREFIX>/run_id=*/jsonl_llm/*.jsonl
# Writes: gs://<bucket>/<STRUCTURED_PREFIX>/datasets/listings_master_llm.csv  (atomic publish)

import csv
import io
import json 
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME        = os.getenv("GCS_BUCKET")                      # REQUIRED
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured") # e.g., "structured"
MAX_FILES= int(os.getenv("MAX_FILES","10"))

storage_client = storage.Client()

# Accept BOTH runIDs:
RUN_ID_ISO_RE= re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE= re.compile(r"^\d{14}$")        # 20251026170002

# Stable CSV schema for students
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage",
    "title_status", "transmission","condition", "fuel", "color", "body_type",
    "city", "state", "zip_code",
    "source_txt", "llm_provider", "llm_model", "llm_ts"
]

def _list_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:  # populate it.prefixes
        pass
    run_ids = [] 
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]           # e.g. run_id=20251026170002
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(rid) or RUN_ID_PLAIN_RE.match(rid):
                run_ids.append(rid)
    return sorted(run_ids)

def _jsonl_records_for_run(bucket:str,structured_prefix: str, run_id: str,max_files: int = 30):
    """
    Yield dict records from the most recent .jsonl files under
    .../run_id=<run_id>/jsonl_llm/
    """
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm/"

    blobs = [blob for blob in b.list_blobs(prefix=prefix) if blob.name.endswith(".jsonl")]

    blobs = sorted(
        blobs,
        key=lambda blob: blob.updated if blob.updated is not None else datetime.now(timezone.utc),
        reverse=True,
    )

    blobs = blobs[:max_files]

    for blob in blobs:
        data = blob.download_as_text()
        line = data.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            rec.setdefault("run_id", run_id)
            yield rec
        except Exception:
            continue

def _run_id_to_dt(rid: str) -> datetime:
    if RUN_ID_ISO_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if RUN_ID_PLAIN_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    # fallback: now
    return datetime.now(timezone.utc)

def _open_gcs_text_writer(bucket: str, key: str):
    """Open a text-mode writer to GCS; close() will finalize the upload."""
    b = storage_client.bucket(bucket)
    blob = b.blob(key)
    # Text mode avoids the flush/finalize pitfall of binary+TextIOWrapper
    return blob.open("w")  # newline handled by csv module


def _write_csv(records: Iterable[Dict], dest_key: str, columns=CSV_COLUMNS) -> int:
    n = 0
    with _open_gcs_text_writer(BUCKET_NAME, dest_key) as out:
        w = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            row = {c: rec.get(c, None) for c in columns}
            w.writerow(row)
            n += 1
    return n  # close() finalizes the upload
    
def _read_existing_master(bucket: str, key: str) -> Dict[str, Dict]:
    """
    Read the existing master CSV from GCS, if it exists, and return a dict
    keyed by post_id.
    """
    b = storage_client.bucket(bucket)
    blob = b.blob(key)

    if not blob.exists():
        return {}

    data = blob.download_as_text()
    reader = csv.DictReader(io.StringIO(data))

    existing = {}
    for row in reader:
        pid = row.get("post_id")
        if pid:
            existing[pid] = row
    return existing
    
def materialize_http(request: Request):
    """
    HTTP POST (no body needed).
    Crawls ALL structured run folders, de-dupes by post_id (keep newest run),
    and writes one CSV directly to .../datasets/listings_master_llm.csv.
    Returns JSON with counts and output path.
    """
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500
                
        try:
            body = request.get_json(silent=True) or {}
        except Exception:
            body = {}

        requested_run_id = body.get("run_id")
        max_files = int(body.get("max_files",MAX_FILES))

        if requested_run_id:
            run_ids = [requested_run_id]
        else:
            run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
            if not run_ids:
                return jsonify({"ok": False, "error": f"no runs found under {STRUCTURED_PREFIX}/"}), 200
            run_ids = run_ids[-1:]   # latest run only
            
        #try:
            #body = request.get_json(silent=True) or {}
        #except Exception:
            #body = {}

        #requested_run_id = body.get("run_id")

        #if requested_run_id:
            #run_ids = [requested_run_id]
        #else:
            #run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
            #if not run_ids:
                #return jsonify({"ok": False, "error": f"no runs found under {STRUCTURED_PREFIX}/"}), 200
            #run_ids = run_ids[-1:]
       base = f"{STRUCTURED_PREFIX}/datasets"
       final_key = f"{base}/listings_master_llm.csv"

# Start from existing master so the dataset grows over time
       latest_by_post: Dict[str, Dict] = _read_existing_master(BUCKET_NAME, final_key)

# Merge in records from the newest run(s)
        for rid in run_ids:
            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid, max_files=max_files):
                pid = rec.get("post_id")
                if not pid:
                    continue
                prev = latest_by_post.get(pid)
                if (prev is None) or (_run_id_to_dt(rec.get("run_id", rid)) > _run_id_to_dt(prev.get("run_id", ""))):
                    latest_by_post[pid] = rec

        rows = _write_csv(latest_by_post.values(), final_key) 
    
       # latest_by_post: Dict[str, Dict] = {}
       # for rid in run_ids:
       #     for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid,max_files=max_files):
       #         pid = rec.get("post_id")
       #         if not pid:
       #             continue
       #         prev = latest_by_post.get(pid)
       #         if (prev is None) or (_run_id_to_dt(rec.get("run_id", rid)) > _run_id_to_dt(prev.get("run_id", ""))):
       #             latest_by_post[pid] = rec

       # base = f"{STRUCTURED_PREFIX}/datasets"
       # final_key = f"{base}/listings_master_llm.csv"
       # rows = _write_csv(latest_by_post.values(), final_key)

        return jsonify({
            "ok": True,
            "runs_scanned": len(run_ids),
            "unique_listings": len(latest_by_post),
            "rows_written": rows,
            "latest_run_processed": run_ids[-1] if run_ids else None,
            #"max_files":max_files,
            "output_csv": f"gs://{BUCKET_NAME}/{final_key}"
        }), 200
    except Exception as e:
        # Return a JSON error so you don't just see a plain 500
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
