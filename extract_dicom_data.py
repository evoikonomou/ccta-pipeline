#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized DICOM header extraction: one row per SeriesInstanceUID.
Processes multiple study folders in parallel.
"""

import os
import sys
import json
import math
import argparse
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from functools import lru_cache

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.tag import BaseTag
from pydicom.datadict import keyword_for_tag, dictionary_description

# -------- Defaults --------
DEFAULT_ROOT = "/home/eo287/mnt/s3_ccta/cta_09232025/studies"
DEFAULT_OUT = "/home/eo287/mnt/s3_ccta/summaries/ccta_series_headers_only.csv"

# Serialization options
FLATTEN_SEQUENCES: bool = True
MAX_SEQ_DEPTH: int = 2
MAX_STRING_LEN: int = 2000
INCLUDE_PRIVATE: bool = True
EXCLUDE_PIXELDATA: bool = True

# -------- Optimized Helpers --------

def likely_dicom_path(fn: str) -> bool:
    """Quick filename filter - accept most files but skip obvious non-DICOM"""
    fn_lower = fn.lower()
    if fn_lower.endswith(('.txt', '.csv', '.json', '.xml', '.log', '.md', '.pdf', '.zip', '.tar', '.gz')):
        return False
    return True

def quick_dicom_check(path: str) -> bool:
    """Fast magic number check without full parse"""
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False

def get_study_folders(root: str) -> List[str]:
    """Get list of study folders (subdirectories of root)"""
    try:
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        # Get immediate subdirectories (study folders like E100138698)
        return [str(p) for p in root_path.iterdir() if p.is_dir()]
    except Exception as e:
        print(f"Error scanning root directory: {e}")
        return []

def iter_all_files_parallel(folder: str, threads: int) -> List[str]:
    """
    Parallel directory traversal - much faster on S3/FUSE.
    Breadth-first traversal with parallel directory scanning.
    """
    paths = []
    dirs_to_scan = [Path(folder)]
    
    def scan_dir(dirpath: Path) -> Tuple[List[str], List[Path]]:
        try:
            entries = list(dirpath.iterdir())
            files = [str(e) for e in entries if e.is_file() and likely_dicom_path(e.name)]
            subdirs = [e for e in entries if e.is_dir()]
            return files, subdirs
        except Exception as e:
            return [], []
    
    while dirs_to_scan:
        with ThreadPoolExecutor(max_workers=min(threads, len(dirs_to_scan))) as ex:
            results = list(ex.map(scan_dir, dirs_to_scan))
        
        dirs_to_scan = []
        for files, subdirs in results:
            paths.extend(files)
            dirs_to_scan.extend(subdirs)
    
    return paths

def batch_by_directory(paths: List[str]) -> Dict[str, List[str]]:
    """Group files by directory for locality-optimized processing"""
    by_dir = defaultdict(list)
    for p in paths:
        by_dir[os.path.dirname(p)].append(p)
    return by_dir

@lru_cache(maxsize=10000)
def tag_to_key_cached(tag_int: int) -> str:
    """Cached tag-to-keyword lookup"""
    tag = BaseTag(tag_int)
    kw = keyword_for_tag(tag)
    if kw:
        return kw
    try:
        name = dictionary_description(tag)
        if name:
            return name.replace(" ", "")
    except KeyError:
        pass
    return f"Private_({tag_int:08X})"

def _truncate(v: str) -> str:
    if v is None:
        return "NA"
    if isinstance(v, str) and len(v) > MAX_STRING_LEN:
        return v[:MAX_STRING_LEN] + "...<truncated>"
    return str(v)

def element_to_value(elem: pydicom.dataelem.DataElement, depth: int = 0) -> Any:
    """Convert DICOM element to serializable value"""
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
    """Convert DICOM dataset to flat dictionary"""
    out: Dict[str, Any] = {}
    for elem in ds:
        if not INCLUDE_PRIVATE and elem.tag.is_private:
            continue
        
        key = elem.keyword or tag_to_key_cached(int(elem.tag))
        try:
            out[key] = element_to_value(elem, depth=depth)
        except Exception:
            out[key] = "<serialization_error>"
    
    return out

def read_header(path: str, defer_size: Optional[str]) -> Optional[pydicom.dataset.Dataset]:
    """Read DICOM header (no pixel data)"""
    try:
        return pydicom.dcmread(
            path, 
            stop_before_pixels=True, 
            force=True, 
            defer_size=defer_size
        )
    except (InvalidDicomError, Exception):
        return None

# -------- Processing --------

def process_study_folder(
    folder: str, 
    threads: int, 
    defer_size: Optional[str],
    magic_check: bool,
    global_seen_series: Dict[str, str],
    global_lock: threading.Lock
) -> List[Dict[str, Any]]:
    """
    Process a single study folder.
    Returns list of row dictionaries.
    """
    
    study_name = os.path.basename(folder.rstrip(os.sep))
    print(f"\n[{study_name}] Scanning directory structure...")
    
    all_paths = iter_all_files_parallel(folder, threads=threads)
    print(f"[{study_name}] Found {len(all_paths):,} potential DICOM files")
    
    if not all_paths:
        return []
    
    batched = batch_by_directory(all_paths)
    print(f"[{study_name}] Files span {len(batched)} directories")
    
    rows: List[Dict[str, Any]] = []
    local_lock = threading.Lock()
    
    processed_count = 0
    skipped_count = 0
    
    def process_file(path: str) -> None:
        nonlocal processed_count, skipped_count
        
        if magic_check and not quick_dicom_check(path):
            with local_lock:
                skipped_count += 1
            return
        
        ds = read_header(path, defer_size)
        if ds is None:
            with local_lock:
                skipped_count += 1
            return
        
        siuid = getattr(ds, "SeriesInstanceUID", None)
        if not siuid:
            with local_lock:
                skipped_count += 1
            return
        
        siuid = str(siuid)
        
        # Check global seen series (across all studies)
        with global_lock:
            if siuid in global_seen_series:
                processed_count += 1
                return
            global_seen_series[siuid] = path
        
        with local_lock:
            processed_count += 1
        
        # Extract header
        hdr = dataset_to_dict(ds, depth=0)
        
        row = {
            "StudyFolder": study_name,
            "SeriesInstanceUID": siuid,
            "RepresentativeFile": path,
        }
        row.update(hdr)
        
        with local_lock:
            rows.append(row)
            if processed_count % 1000 == 0:
                print(f"[{study_name}] Processed {processed_count:,} files...")
    
    print(f"[{study_name}] Processing with {threads} threads...")
    
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = []
        for dir_path, files in batched.items():
            for fpath in files:
                futures.append(ex.submit(process_file, fpath))
        
        for fut in as_completed(futures):
            fut.result()
    
    print(f"[{study_name}] Completed: {processed_count:,} processed, {skipped_count:,} skipped, {len(rows)} unique series")
    
    return rows

def process_all_studies(
    root: str,
    threads: int,
    defer_size: Optional[str],
    magic_check: bool,
    study_parallel: int = 4
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Process all study folders under root directory.
    Processes multiple studies in parallel for better efficiency.
    """
    
    study_folders = get_study_folders(root)
    
    if not study_folders:
        print(f"ERROR: No study folders found in {root}")
        return [], set()
    
    print(f"Found {len(study_folders)} study folders to process")
    print(f"Study folders: {[os.path.basename(f) for f in study_folders[:10]]}" + 
          (f" ... and {len(study_folders)-10} more" if len(study_folders) > 10 else ""))
    print(f"Processing {study_parallel} studies in parallel, {threads} threads per study")
    print()
    
    # Global tracking across all studies
    global_seen_series: Dict[str, str] = {}
    global_lock = threading.Lock()
    all_rows: List[Dict[str, Any]] = []
    union_keys: Set[str] = set()
    results_lock = threading.Lock()
    
    completed_count = 0
    
    def process_and_collect(study_folder: str) -> None:
        nonlocal completed_count
        
        study_rows = process_study_folder(
            study_folder,
            threads=threads,
            defer_size=defer_size,
            magic_check=magic_check,
            global_seen_series=global_seen_series,
            global_lock=global_lock
        )
        
        # Collect results
        with results_lock:
            all_rows.extend(study_rows)
            for row in study_rows:
                union_keys.update(row.keys())
            completed_count += 1
            print(f"Progress: {completed_count}/{len(study_folders)} studies completed")
    
    # Process studies in parallel
    with ThreadPoolExecutor(max_workers=study_parallel) as ex:
        futures = [ex.submit(process_and_collect, sf) for sf in study_folders]
        for fut in as_completed(futures):
            fut.result()  # Raise any exceptions
    
    print(f"\n{'='*60}")
    print(f"TOTAL: Processed {len(study_folders)} studies, found {len(all_rows)} unique series")
    print(f"{'='*60}\n")
    
    return all_rows, union_keys

# -------- Main --------

def main():
    global MAX_SEQ_DEPTH, INCLUDE_PRIVATE, EXCLUDE_PIXELDATA
    
    ap = argparse.ArgumentParser(
        description="Optimized DICOM header extraction: one row per SeriesInstanceUID across multiple studies"
    )
    ap.add_argument("--root", default=DEFAULT_ROOT, 
                    help="Path to studies root folder (containing study subfolders)")
    ap.add_argument("--out", default=DEFAULT_OUT, 
                    help="Output CSV path")
    ap.add_argument("--threads", type=int, default=8, 
                    help="Parallel threads per study (16-64 recommended for S3/FUSE)")
    ap.add_argument("--study-parallel", type=int, default=8,
                    help="Number of studies to process in parallel (2-8 recommended)")
    ap.add_argument("--defer-size", default="64 KB",
                    help="Defer reading elements larger than this")
    ap.add_argument("--max-seq-depth", type=int, default=MAX_SEQ_DEPTH, 
                    help="Sequence flattening depth")
    ap.add_argument("--include-private", action="store_true", default=INCLUDE_PRIVATE, 
                    help="Include private tags")
    ap.add_argument("--keep-pixeldata", action="store_true", 
                    help="Include PixelData (not recommended)")
    ap.add_argument("--no-magic-check", action="store_true",
                    help="Skip fast DICM magic number check")
    args = ap.parse_args()
    
    MAX_SEQ_DEPTH = args.max_seq_depth
    INCLUDE_PRIVATE = args.include_private
    EXCLUDE_PIXELDATA = not args.keep_pixeldata
    
    defer_size: Optional[str]
    if str(args.defer_size).strip() in ("0", "0B", "0KB", "0MB"):
        defer_size = None
    else:
        defer_size = args.defer_size
    
    root = args.root
    out_csv = args.out
    
    if not os.path.isdir(root):
        print(f"ERROR: root folder not found: {root}", file=sys.stderr)
        sys.exit(2)
    
    print(f"Extracting DICOM headers from studies in: {root}")
    print(f"Output: {out_csv}")
    print(f"Study parallelism: {args.study_parallel}")
    print(f"Threads per study: {args.threads}")
    print(f"Defer size: {defer_size}")
    print(f"Magic check: {not args.no_magic_check}")
    print()
    
    rows, keys = process_all_studies(
        root, 
        threads=args.threads, 
        defer_size=defer_size,
        magic_check=not args.no_magic_check,
        study_parallel=args.study_parallel
    )
    
    if not rows:
        print("No series found; nothing to write.")
        sys.exit(0)
    
    # Build DataFrame with ordered columns
    first_cols = ["StudyFolder", "SeriesInstanceUID", "RepresentativeFile"]
    all_cols = first_cols + sorted(k for k in keys if k not in first_cols)
    df = pd.DataFrame(rows, columns=all_cols)
    
    # Write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    print(f"✓ Wrote {len(df):,} series headers to: {out_csv}")
    print()
    print("Preview:")
    with pd.option_context("display.max_columns", 20, "display.width", 220):
        print(df.head(10))
    
    # Show breakdown by study
    print("\nSeries count by study:")
    print(df['StudyFolder'].value_counts().head(20))

if __name__ == "__main__":
    main()