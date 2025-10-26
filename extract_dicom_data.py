#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract one DICOM header per SeriesInstanceUID within a single study folder (e.g., /studies/E100138698).
- Recursively walks the study directory
- Picks exactly one representative file per unique SeriesInstanceUID (volume)
- Writes one row per volume; columns are flattened DICOM header elements (no PixelData)
"""

import os
import sys
import json
import math
import argparse
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.tag import BaseTag
from pydicom.datadict import keyword_for_tag, dictionary_description

# -------- Defaults --------
DEFAULT_ROOT = "/mnt/s3_ccta/cta_09232025/studies/E100138698"
DEFAULT_OUT  = "/mnt/s3_ccta/summaries/ccta_series_headers_only_E100138698.csv"

# Serialization options
FLATTEN_SEQUENCES: bool = True
MAX_SEQ_DEPTH: int = 2
MAX_STRING_LEN: int = 2000
INCLUDE_PRIVATE: bool = True
EXCLUDE_PIXELDATA: bool = True

FAST_TAGS = ["SeriesInstanceUID"]  # minimal read to identify unique series

# -------- Helpers --------
def likely_dicom_path(fn: str) -> bool:
    """Be permissive: accept any filename; many exports are 1.dcm/2.dcm or no extension."""
    # If you want to be stricter, check extensions or sniff 'DICM' magic.
    return True

def read_fast(path: str, defer_size: Optional[str]):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True,
                               specific_tags=FAST_TAGS, defer_size=defer_size)
    except (InvalidDicomError, Exception):
        return None

def read_full_header(path: str, defer_size: Optional[str]):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True, defer_size=defer_size)
    except (InvalidDicomError, Exception):
        return None

def tag_to_key(tag: BaseTag) -> str:
    kw = keyword_for_tag(tag)
    if kw:
        return kw
    name = dictionary_description(tag)
    if name:
        return name.replace(" ", "")
    return f"Private_({int(tag):08X})"

def _truncate(v: str) -> str:
    if v is None:
        return "NA"
    if isinstance(v, str) and len(v) > MAX_STRING_LEN:
        return v[:MAX_STRING_LEN] + "...<truncated>"
    return str(v)

def element_to_value(elem: pydicom.dataelem.DataElement, depth: int = 0) -> Any:
    if EXCLUDE_PIXELDATA and elem.keyword == "PixelData":
        return "<PixelData omitted>"
    val = elem.value
    if elem.VR == "SQ" and FLATTEN_SEQUENCES:
        if depth >= MAX_SEQ_DEPTH:
            try:
                return _truncate(json.dumps([f"Item{idx}" for idx, _ in enumerate(val)], ensure_ascii=False))
            except Exception:
                return f"<SQ depth>{len(val)} items"
        items = []
        for it in val:
            items.append(dataset_to_dict(it, depth=depth + 1))
        try:
            return _truncate(json.dumps(items, ensure_ascii=False))
        except Exception:
            return _truncate(str(items))
    if isinstance(val, (list, tuple)):
        return _truncate("|".join(_truncate(str(x)) for x in val))
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "NA"
    return _truncate(str(val))

def dataset_to_dict(ds: pydicom.dataset.Dataset, depth: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for elem in ds:
        if not INCLUDE_PRIVATE and elem.tag.is_private:
            continue
        key = elem.keyword or tag_to_key(elem.tag)
        try:
            out[key] = element_to_value(elem, depth=depth)
        except Exception:
            out[key] = "<serialization_error>"
    return out

def iter_all_files(folder: str) -> List[str]:
    paths = []
    for dirpath, _, files in os.walk(folder):
        for fn in files:
            if likely_dicom_path(fn):
                paths.append(os.path.join(dirpath, fn))
    return paths

def discover_representatives(folder: str, threads: int, defer_size: Optional[str]) -> Dict[str, str]:
    """
    Return { SeriesInstanceUID: representative_file_path } discovered via a fast pass.
    """
    all_paths = iter_all_files(folder)
    rep: Dict[str, str] = {}

    def probe(path: str):
        ds = read_fast(path, defer_size)
        if ds is None:
            return None
        siuid = getattr(ds, "SeriesInstanceUID", None)
        if not siuid:
            return None
        return str(siuid), path

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for fut in as_completed([ex.submit(probe, p) for p in all_paths]):
            res = fut.result()
            if res is None:
                continue
            siuid, path = res
            # keep the first file seen per series
            if siuid not in rep:
                rep[siuid] = path
    return rep

def serialize_headers(rep: Dict[str, str], folder: str, threads: int, defer_size: Optional[str]) -> Tuple[List[Dict[str, Any]], set]:
    rows: List[Dict[str, Any]] = []
    union_keys: set = set()

    def read_one(siuid: str, fpath: str):
        ds_full = read_full_header(fpath, defer_size)
        if ds_full is None:
            return None
        hdr = dataset_to_dict(ds_full, depth=0)
        row = {
            "StudyFolder": os.path.basename(folder.rstrip(os.sep)),
            "SeriesInstanceUID": siuid,
            "RepresentativeFile": fpath,
        }
        row.update(hdr)
        return row

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(read_one, s, p) for s, p in rep.items()]
        for fut in as_completed(futs):
            row = fut.result()
            if row is None:
                continue
            rows.append(row)
            union_keys.update(row.keys())

    return rows, union_keys

# -------- Main --------
def main():
    global MAX_SEQ_DEPTH, INCLUDE_PRIVATE, EXCLUDE_PIXELDATA

    ap = argparse.ArgumentParser(description="Extract one header per SeriesInstanceUID for a single study folder.")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Path to a single study (e.g., /studies/E100138698)")
    ap.add_argument("--out",  default=DEFAULT_OUT,  help="Output CSV path")
    ap.add_argument("--threads", type=int, default=32, help="I/O threads (16–64 is reasonable on S3/FUSE)")
    ap.add_argument("--defer-size", default="64 KB",
                    help="Defer reading elements larger than this (e.g., '64 KB', '1 MB'; use '0' to disable)")
    ap.add_argument("--max-seq-depth", type=int, default=MAX_SEQ_DEPTH, help="Sequence flattening depth")
    ap.add_argument("--include-private", action="store_true", default=INCLUDE_PRIVATE, help="Include private tags")
    ap.add_argument("--keep-pixeldata", action="store_true", help="Include PixelData (not recommended)")
    args, _ = ap.parse_known_args()

    # Runtime config
    MAX_SEQ_DEPTH     = args.max_seq_depth
    INCLUDE_PRIVATE   = args.include_private
    EXCLUDE_PIXELDATA = not args.keep_pixeldata

    # pydicom accepts int bytes or strings like '64 KB'; we pass through unless '0'
    defer_size: Optional[str]
    if str(args.defer_size).strip() in ("0", "0B", "0KB", "0MB"):
        defer_size = None
    else:
        defer_size = args.defer_size

    folder = args.root
    out_csv = args.out

    if not os.path.isdir(folder):
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    # 1) discover unique SeriesInstanceUID -> representative file
    rep = discover_representatives(folder, threads=args.threads, defer_size=defer_size)
    if not rep:
        print("No series found; nothing to write.")
        sys.exit(0)

    # 2) read headers for those representative files
    rows, keys = serialize_headers(rep, folder=folder, threads=args.threads, defer_size=defer_size)

    # 3) write CSV
    first_cols = ["StudyFolder", "SeriesInstanceUID", "RepresentativeFile"]
    df = pd.DataFrame(rows, columns=first_cols + sorted(k for k in keys if k not in first_cols))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Wrote {len(df):,} series headers to: {out_csv}")
    with pd.option_context("display.max_columns", 20, "display.width", 220):
        print(df.head(5))

if __name__ == "__main__":
    main()
