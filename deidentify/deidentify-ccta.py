#%%
#!/usr/bin/env python3
"""
De-identify CT DICOM series from an input folder, keeping only true volumes and
dropping likely burned-in / secondary capture content.

Outputs:
  - Deidentified DICOMs:
      <out_root>/<EID>-deid/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
  - Per-case summary:
      <out_root>/<EID>-deid/summary.txt
  - Logs/metadata:
      <log_root>/<EID>-metadata.csv
      <log_root>/<EID>-metadata.txt

Usage example:
  python deid_ct_volumes.py \
    --input /home/eo287/mnt/s3_ccta/cta_09232025/studies/E100138698/1.2.840.113845.11.1000000001799338748.20130201075411.7339398 \
    --eid E100138698 \
    --out-root /home/eo287/mnt/s3_ccta/deidentified \
    --log-root /home/eo287/mnt/s3_ccta/deidentified-logs \
    --min-slices 16 \
    --workers 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pydicom
from pydicom import dcmread
from pydicom.uid import generate_uid, UID
from pydicom.errors import InvalidDicomError


# ----------------------------
# Configuration / heuristics
# ----------------------------

# A pragmatic PHI tag list (not exhaustive, but broad). We also remove private tags.
# If you need strict Basic Application Confidentiality Profile compliance, consider
# integrating a dedicated de-id library/policy engine and formally documenting it.
PHI_TAGS_TO_BLANK = [
    # Patient / person identifiers
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0032),  # PatientBirthTime
    (0x0010, 0x0040),  # PatientSex (may keep; depends; here we blank)
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0010, 0x1040),  # PatientAddress
    (0x0010, 0x1060),  # PatientMotherBirthName
    (0x0010, 0x2154),  # PatientTelephoneNumbers
    (0x0010, 0x2160),  # EthnicGroup (often considered sensitive)
    (0x0010, 0x2180),  # Occupation
    (0x0010, 0x21B0),  # AdditionalPatientHistory
    (0x0010, 0x4000),  # PatientComments

    # Study/series dates & times (often PHI depending on policy)
    (0x0008, 0x0020),  # StudyDate
    (0x0008, 0x0021),  # SeriesDate
    (0x0008, 0x0022),  # AcquisitionDate
    (0x0008, 0x0023),  # ContentDate
    (0x0008, 0x0030),  # StudyTime
    (0x0008, 0x0031),  # SeriesTime
    (0x0008, 0x0032),  # AcquisitionTime
    (0x0008, 0x0033),  # ContentTime

    # Accession / institution / operators
    (0x0008, 0x0050),  # AccessionNumber
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x0092),  # ReferringPhysicianAddress
    (0x0008, 0x0094),  # ReferringPhysicianTelephoneNumbers
    (0x0008, 0x1010),  # StationName
    (0x0008, 0x1040),  # InstitutionalDepartmentName
    (0x0008, 0x1070),  # OperatorsName

    # Device / site-ish identifiers (policy dependent)
    (0x0018, 0x1000),  # DeviceSerialNumber
    (0x0018, 0x1002),  # DeviceUID

    # Other identifiers
    (0x0040, 0x0241),  # PerformedStationAETitle
    (0x0040, 0x0242),  # PerformedStationName
    (0x0040, 0x0243),  # PerformedLocation
    (0x0040, 0x0254),  # PerformedProcedureStepDescription
    (0x0040, 0x0275),  # RequestAttributesSequence (may contain identifiers; we remove below if present)
]

# Sequences that often contain identifiers; safest to delete entirely.
SEQUENCES_TO_DELETE = [
    (0x0008, 0x1110),  # ReferencedStudySequence
    (0x0008, 0x1111),  # ReferencedPerformedProcedureStepSequence
    (0x0008, 0x1115),  # ReferencedSeriesSequence
    (0x0008, 0x1120),  # ReferencedPatientSequence
    (0x0008, 0x1125),  # ReferencedVisitSequence
    (0x0008, 0x1140),  # ReferencedImageSequence
    (0x0010, 0x1002),  # OtherPatientIDsSequence
    (0x0038, 0x0004),  # ReferencedPatientAliasSequence
    (0x0040, 0x0275),  # RequestAttributesSequence
    (0x0040, 0xA730),  # ContentSequence (can embed names in SR-like objects; we exclude non-CT anyway)
]


@dataclass
class DicomFileInfo:
    path: str
    study_uid: str
    series_uid: str
    sop_uid: str
    modality: str
    sop_class_uid: str
    series_desc: str
    image_type: str
    burned_in: str
    rows: Optional[int]
    cols: Optional[int]
    pixel_spacing: str
    slice_thickness: str
    spacing_between_slices: str
    image_position_patient: str
    image_orientation_patient: str
    instance_number: Optional[int]


@dataclass
class SeriesSummary:
    eid: str
    study_uid_out: str
    series_uid_out: str
    series_desc: str
    n_slices: int
    rows: Optional[int]
    cols: Optional[int]
    pixel_spacing: str
    slice_thickness: str
    spacing_between_slices: str
    dropped_reason: str  # empty if kept
    source_series_uid: str
    source_study_uid: str


class UIDMapper:
    """
    Deterministic UID mapping per run (consistent within an execution).
    If you want stable mappings across multiple runs, seed with a stable salt and persist map.
    """
    def __init__(self, salt: str):
        self.salt = salt
        self.map: Dict[str, str] = {}

    def __call__(self, uid: str) -> str:
        if uid in self.map:
            return self.map[uid]
        # generate a deterministic UID using a hash -> numeric string
        h = hashlib.sha256((self.salt + uid).encode("utf-8")).hexdigest()
        # DICOM UID must be digits and dots. Use a prefix OID + int from hash.
        # 2.25.<decimal> is valid: "2.25" + integer representation of UUID-like.
        dec = int(h[:32], 16)  # 128-bit chunk
        new_uid = f"2.25.{dec}"
        self.map[uid] = new_uid
        return new_uid


def is_probably_dicom(path: Path) -> bool:
    # Fast check: DICOM magic "DICM" at byte 128 for standard files,
    # but many valid DICOMs are missing it. We'll try a cheap open/read.
    try:
        with path.open("rb") as f:
            pre = f.read(132)
        return len(pre) >= 132 and pre[128:132] == b"DICM"
    except Exception:
        return False


def safe_get(ds, name: str, default="") -> str:
    v = getattr(ds, name, default)
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        return "\\".join(str(x) for x in v)
    return str(v)


def read_header_minimal(path: Path) -> Optional[DicomFileInfo]:
    try:
        ds = dcmread(str(path), stop_before_pixels=True, force=True, specific_tags=[
            "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
            "Modality", "SOPClassUID", "SeriesDescription", "ImageType",
            "BurnedInAnnotation", "Rows", "Columns", "PixelSpacing",
            "SliceThickness", "SpacingBetweenSlices", "ImagePositionPatient",
            "ImageOrientationPatient", "InstanceNumber"
        ])
    except InvalidDicomError:
        return None
    except Exception:
        return None

    study_uid = safe_get(ds, "StudyInstanceUID", "")
    series_uid = safe_get(ds, "SeriesInstanceUID", "")
    sop_uid = safe_get(ds, "SOPInstanceUID", "")
    modality = safe_get(ds, "Modality", "")
    sop_class_uid = safe_get(ds, "SOPClassUID", "")
    series_desc = safe_get(ds, "SeriesDescription", "")
    image_type = safe_get(ds, "ImageType", "")
    burned_in = safe_get(ds, "BurnedInAnnotation", "")
    rows = getattr(ds, "Rows", None)
    cols = getattr(ds, "Columns", None)
    pixel_spacing = safe_get(ds, "PixelSpacing", "")
    slice_thickness = safe_get(ds, "SliceThickness", "")
    spacing_between_slices = safe_get(ds, "SpacingBetweenSlices", "")
    ipp = safe_get(ds, "ImagePositionPatient", "")
    iop = safe_get(ds, "ImageOrientationPatient", "")
    inst = getattr(ds, "InstanceNumber", None)

    if not (study_uid and series_uid and sop_uid):
        return None

    return DicomFileInfo(
        path=str(path),
        study_uid=study_uid,
        series_uid=series_uid,
        sop_uid=sop_uid,
        modality=modality,
        sop_class_uid=sop_class_uid,
        series_desc=series_desc,
        image_type=image_type,
        burned_in=burned_in,
        rows=rows,
        cols=cols,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
        spacing_between_slices=spacing_between_slices,
        image_position_patient=ipp,
        image_orientation_patient=iop,
        instance_number=inst,
    )


def series_drop_reason(files: List[DicomFileInfo], min_slices: int) -> str:
    # Must be CT
    modalities = {f.modality.upper() for f in files}
    if modalities != {"CT"}:
        return f"non-CT modality set={modalities}"

    # Burned-in annotation explicit flag
    # 0028,0301 values: "YES"/"NO" (can be absent)
    if any(f.burned_in.strip().upper() == "YES" for f in files):
        return "BurnedInAnnotation=YES"

    # ImageType heuristic: drop SECONDARY/DERIVED and common screenshot-like series
    # ImageType can be like: ORIGINAL\PRIMARY\AXIAL or DERIVED\SECONDARY\...
    bad_tokens = {"SECONDARY", "DERIVED", "SCREEN", "LOCALIZER"}
    for f in files:
        toks = {t.strip().upper() for t in f.image_type.split("\\") if t.strip()}
        if toks & bad_tokens:
            return f"ImageType contains {sorted(toks & bad_tokens)}"

    # Minimum slice count (volume)
    if len(files) < min_slices:
        return f"too few slices (n={len(files)} < {min_slices})"

    # Geometry consistency checks (basic)
    # Ensure Rows/Cols consistent and iop present
    rc = {(f.rows, f.cols) for f in files}
    if len(rc) != 1:
        return f"inconsistent matrix sizes: {sorted(rc)}"

    iops = {f.image_orientation_patient for f in files if f.image_orientation_patient}
    if len(iops) > 1:
        return "inconsistent ImageOrientationPatient"

    # PixelSpacing consistency
    pss = {f.pixel_spacing for f in files if f.pixel_spacing}
    if len(pss) > 1:
        return "inconsistent PixelSpacing"

    # Many non-volume objects have missing IPP/InstanceNumber; require at least some ordering info
    n_has_ipp = sum(1 for f in files if f.image_position_patient)
    n_has_inst = sum(1 for f in files if f.instance_number is not None)
    if n_has_ipp < max(3, len(files) // 10) and n_has_inst < max(3, len(files) // 10):
        return "insufficient slice position info (IPP/InstanceNumber largely missing)"

    return ""


def sort_series(files: List[DicomFileInfo]) -> List[DicomFileInfo]:
    # Prefer z-position ordering from IPP; fallback to InstanceNumber; else filename
    def parse_ipp_z(ipp: str) -> Optional[float]:
        try:
            parts = [float(x) for x in ipp.replace(",", " ").split()]
            if len(parts) == 3:
                return parts[2]
        except Exception:
            return None
        return None

    with_z = [(parse_ipp_z(f.image_position_patient), f) for f in files]
    if sum(z is not None for z, _ in with_z) >= len(files) * 0.7:
        return [f for _, f in sorted(with_z, key=lambda t: (t[0] is None, t[0]))]

    if sum(f.instance_number is not None for f in files) >= len(files) * 0.7:
        return sorted(files, key=lambda f: (f.instance_number is None, f.instance_number))

    return sorted(files, key=lambda f: f.path)


def deidentify_dataset(ds: pydicom.dataset.Dataset, eid: str, uid_map: UIDMapper) -> pydicom.dataset.Dataset:
    """
    In-place de-id of a DICOM dataset. Returns ds.
    IMPORTANT: This does not scrub burned-in pixel PHI. We attempt to drop likely
    burned-in series; for robust pixel redaction you'd need a dedicated pipeline.
    """

    # Remove private tags (vendor-specific; can contain identifiers)
    ds.remove_private_tags()

    # Blank common PHI tags if present
    for tag in PHI_TAGS_TO_BLANK:
        if tag in ds:
            ds[tag].value = ""

    # Delete risky sequences wholesale
    for tag in SEQUENCES_TO_DELETE:
        if tag in ds:
            del ds[tag]

    # Replace patient module with de-id tokens
    ds.PatientName = f"{eid}-DEID"
    ds.PatientID = f"{eid}-DEID"
    ds.PatientBirthDate = ""
    ds.PatientSex = ""

    # De-id any “Other” name-like fields that commonly appear
    for attr in [
        "ReferringPhysicianName",
        "OperatorsName",
        "PerformingPhysicianName",
        "InstitutionName",
        "StationName",
    ]:
        if hasattr(ds, attr):
            try:
                setattr(ds, attr, "")
            except Exception:
                pass

    # Regenerate UIDs consistently
    if "StudyInstanceUID" in ds:
        ds.StudyInstanceUID = uid_map(str(ds.StudyInstanceUID))
    if "SeriesInstanceUID" in ds:
        ds.SeriesInstanceUID = uid_map(str(ds.SeriesInstanceUID))
    if "SOPInstanceUID" in ds:
        ds.SOPInstanceUID = uid_map(str(ds.SOPInstanceUID))

    # Also update FrameOfReferenceUID if present (often used in CT)
    if "FrameOfReferenceUID" in ds:
        ds.FrameOfReferenceUID = uid_map(str(ds.FrameOfReferenceUID))

    # Ensure file meta UIDs align
    if hasattr(ds, "file_meta") and ds.file_meta is not None:
        if "MediaStorageSOPInstanceUID" in ds.file_meta and "SOPInstanceUID" in ds:
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    # Clear textual comments
    if "ImageComments" in ds:
        ds.ImageComments = ""
    if "PatientComments" in ds:
        ds.PatientComments = ""

    return ds


def write_deid_file(
    src_path: Path,
    dst_path: Path,
    eid: str,
    uid_map: UIDMapper,
) -> Tuple[bool, str]:
    try:
        ds = dcmread(str(src_path), force=True)  # includes pixels
        ds = deidentify_dataset(ds, eid=eid, uid_map=uid_map)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(dst_path), write_like_original=False)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def find_dicom_files(root: Path) -> List[Path]:
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        # Accept .dcm and also extensionless files that look like DICOM
        if p.suffix.lower() == ".dcm" or is_probably_dicom(p):
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Input folder containing DICOMs (recursively).")
    ap.add_argument("--eid", required=True, type=str, help="Case identifier, e.g., E100138698.")
    ap.add_argument("--out-root", required=True, type=str, help="Output root for deidentified cases.")
    ap.add_argument("--log-root", required=True, type=str, help="Output root for logs/metadata.")
    ap.add_argument("--min-slices", type=int, default=16, help="Minimum slices per series to keep as volume.")
    ap.add_argument("--workers", type=int, default=8, help="Threads for writing DICOMs.")
    ap.add_argument("--dry-run", action="store_true", help="Scan and report only; do not write outputs.")
    args = ap.parse_args()

    in_root = Path(args.input)
    eid = args.eid
    out_root = Path(args.out_root)
    log_root = Path(args.log_root)

    case_out = out_root / f"{eid}-deid"
    case_out.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    uid_map = UIDMapper(salt=f"{eid}|{int(time.time())}")  # per-run salt; makes UIDs unlinkable across runs

    # 1) Enumerate files
    dicom_paths = find_dicom_files(in_root)
    if not dicom_paths:
        print(f"[ERROR] No DICOM-like files found under: {in_root}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Found {len(dicom_paths)} candidate DICOM files")

    # 2) Read minimal headers (fast) and group by (StudyUID, SeriesUID)
    infos: List[DicomFileInfo] = []
    n_bad = 0
    for p in dicom_paths:
        info = read_header_minimal(p)
        if info is None:
            n_bad += 1
            continue
        infos.append(info)

    print(f"[INFO] Parsed headers for {len(infos)} files (skipped {n_bad} unreadable)")

    by_series: Dict[Tuple[str, str], List[DicomFileInfo]] = {}
    for inf in infos:
        key = (inf.study_uid, inf.series_uid)
        by_series.setdefault(key, []).append(inf)

    print(f"[INFO] Detected {len(by_series)} series")

    # 3) Decide which series to keep (CT volumes only)
    series_summaries: List[SeriesSummary] = []
    kept_series: List[Tuple[Tuple[str, str], List[DicomFileInfo]]] = []

    for (study_uid, series_uid), files in by_series.items():
        reason = series_drop_reason(files, min_slices=args.min_slices)
        # Compute mapped output UIDs (even for dropped, for consistent reporting)
        study_uid_out = uid_map(study_uid)
        series_uid_out = uid_map(series_uid)

        # representative metadata (from first file)
        f0 = files[0]
        ss = SeriesSummary(
            eid=eid,
            study_uid_out=study_uid_out,
            series_uid_out=series_uid_out,
            series_desc=f0.series_desc,
            n_slices=len(files),
            rows=f0.rows,
            cols=f0.cols,
            pixel_spacing=f0.pixel_spacing,
            slice_thickness=f0.slice_thickness,
            spacing_between_slices=f0.spacing_between_slices,
            dropped_reason=reason,
            source_series_uid=series_uid,
            source_study_uid=study_uid,
        )
        series_summaries.append(ss)

        if reason == "":
            kept_series.append(((study_uid, series_uid), sort_series(files)))

    print(f"[INFO] Keeping {len(kept_series)} series as CT volumes; dropping {len(by_series) - len(kept_series)} series")

    # 4) Write outputs
    # Write metadata logs (CSV + TXT)
    csv_path = log_root / f"{eid}-metadata.csv"
    txt_path = log_root / f"{eid}-metadata.txt"

    fieldnames = list(asdict(series_summaries[0]).keys()) if series_summaries else []
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ss in series_summaries:
            w.writerow(asdict(ss))

    with txt_path.open("w") as f:
        f.write(f"EID: {eid}\n")
        f.write(f"Input: {in_root}\n")
        f.write(f"Total files scanned: {len(dicom_paths)}\n")
        f.write(f"Header-parsed files: {len(infos)}\n")
        f.write(f"Total series: {len(by_series)}\n")
        f.write(f"Kept series (volumes): {len(kept_series)}\n\n")
        for ss in series_summaries:
            status = "KEPT" if ss.dropped_reason == "" else f"DROPPED ({ss.dropped_reason})"
            f.write(
                f"- {status} | src Study={ss.source_study_uid} Series={ss.source_series_uid} "
                f"| out Study={ss.study_uid_out} Series={ss.series_uid_out} "
                f"| n={ss.n_slices} | desc='{ss.series_desc}' | matrix={ss.rows}x{ss.cols} "
                f"| ps={ss.pixel_spacing} | thk={ss.slice_thickness}\n"
            )

    # Per-case summary inside case folder
    summary_path = case_out / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"EID: {eid}\n")
        f.write(f"Deidentified output root: {case_out}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Kept CT volumes: {len(kept_series)}\n\n")
        for ss in [s for s in series_summaries if s.dropped_reason == ""]:
            f.write(
                f"- StudyUID={ss.study_uid_out} SeriesUID={ss.series_uid_out} "
                f"| n_slices={ss.n_slices} | desc='{ss.series_desc}' "
                f"| matrix={ss.rows}x{ss.cols} | PixelSpacing={ss.pixel_spacing} "
                f"| SliceThickness={ss.slice_thickness} | SpacingBetweenSlices={ss.spacing_between_slices}\n"
            )

    if args.dry_run:
        print(f"[DRY-RUN] Wrote logs only:\n  {csv_path}\n  {txt_path}\n  {summary_path}")
        sys.exit(0)

    # 5) Copy & de-id files for kept series
    # Use ThreadPoolExecutor (I/O bound + pydicom parsing). If CPU becomes limiting, swap to ProcessPool.
    failures: List[Tuple[str, str]] = []
    n_written = 0

    tasks = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for (study_uid, series_uid), files in kept_series:
            study_uid_out = uid_map(study_uid)
            series_uid_out = uid_map(series_uid)
            out_dir = case_out / study_uid_out / series_uid_out

            for i, finfo in enumerate(files):
                src = Path(finfo.path)
                # Preserve ordering by naming; avoid leaking original filenames
                dst = out_dir / f"slice_{i:06d}.dcm"
                tasks.append(ex.submit(write_deid_file, src, dst, eid, uid_map))

        for fut in as_completed(tasks):
            ok, err = fut.result()
            if ok:
                n_written += 1
            else:
                failures.append(("write_failed", err))

    # 6) Write failure report
    failure_path = case_out / "failures.txt"
    with failure_path.open("w") as f:
        f.write(f"Total write attempts: {len(tasks)}\n")
        f.write(f"Wrote successfully: {n_written}\n")
        f.write(f"Failures: {len(failures)}\n\n")
        for kind, msg in failures[:5000]:
            f.write(f"- {kind}: {msg}\n")

    print(f"[INFO] Done. Wrote {n_written}/{len(tasks)} DICOMs")
    print(f"[INFO] Case summary: {summary_path}")
    print(f"[INFO] Metadata CSV: {csv_path}")
    print(f"[INFO] Failures: {failure_path}")


if __name__ == "__main__":
    main()
