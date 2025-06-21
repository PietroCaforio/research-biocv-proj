#!/usr/bin/env python3
"""
match_raw_volumes.py
--------------------

Reproduce the original preprocessing pipeline (224×224×224 resample → remap
occupied slices → enforce depth = 66 along axis-0) on every candidate raw CT
series and find those that recreate the reference *.npy arrays contained in a
fold CSV.

Logs *all* matches (patient, index, raw series UID, path, delta).

Assumptions confirmed by the user
---------------------------------
• Saved reference arrays are shaped (66, 224, 224)  — i.e. slices on axis-0.
• oversampling was **NOT** used in the original preprocessing.
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Import your utility helpers ------------------------------------------------ #
sys.path.insert(0, "./")  # adjust if util/ is elsewhere
from util.data_util import (  # noqa: E402
    get_occupied_slices,
    load_single_volume,
    preprocess,
    remap_occupied_slices,
)

TARGET_SHAPE = [224, 224, 224]
FIX_DEPTH = 66
# --- helper (add to top section) -----------------------------------------
import re   # already imported above, kept for clarity

def normpath(p: str) -> str:
    return p.replace("\\", "/") if os.name != "nt" else p

def resolve_ct_path(raw_root: Path, file_loc: str) -> Path:
    parts = Path(normpath(file_loc)).parts
    if parts and (parts[0] in ('.', '') or re.match(r"^[A-Za-z]:$", parts[0])):
        parts = parts[1:]                       # drop '.' or 'C:'
    if parts and parts[0] == raw_root.name:
        parts = parts[1:]                       # drop duplicate 'CPTAC-PDA'
    return raw_root.joinpath(*parts)

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: str = "./logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    logfile = Path(log_dir) / f"match_log_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        filename=logfile,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(console)
    logging.info("Logging initialised.")
    return str(logfile)


# --------------------------------------------------------------------------- #
#  Depth enforcement (pads / trims axis-0) ---------------------------------- #
def enforce_depth(indices: List[int], volume_depth: int) -> List[int]:
    indices = sorted(indices)
    if len(indices) == FIX_DEPTH:
        return indices

    if len(indices) < FIX_DEPTH:  # pad with non-tumour slices
        left, right = indices[0] - 1, indices[-1] + 1
        while len(indices) < FIX_DEPTH:
            if left >= 0:
                indices.insert(0, left)
                left -= 1
            if len(indices) < FIX_DEPTH and right < volume_depth:
                indices.append(right)
                right += 1
        return indices

    # len(indices) > FIX_DEPTH  → keep central 66
    mid = (indices[0] + indices[-1]) // 2
    half = FIX_DEPTH // 2
    start = mid - half
    return list(range(start, start + FIX_DEPTH))


def normpath(p: str) -> str:
    return p.replace("\\", "/") if os.name != "nt" else p


# --------------------------------------------------------------------------- #
#  Worker ------------------------------------------------------------------- #
def worker(task) -> List[Tuple]:
    """
    For one (patient, idx) fold entry test every segmentation-linked raw series.
    Return list of matches: (patient, idx, raw_series_uid, raw_folder, delta).
    """
    patient, idx = task["patient_id"], task["idx"]
    embedding_path = task["embedding_path"]
    ref_np = task["ref_np_path"]
    seg_rows = task["seg_df"]
    raw_meta = task["raw_meta_df"]
    args = task["args"]

    matches: List[Tuple] = []

    # ---------- load reference array -------------------------------------- #
    try:
        ref_arr = np.load(ref_np)
    except FileNotFoundError:
        logging.warning(f"[{patient} | {idx}] reference .npy not found: {ref_np}")
        return matches

    ref_shape = ref_arr.shape  # e.g. (66, 224, 224)

    # ---------- iterate over each segmentation object --------------------- #
    for _, seg in seg_rows.iterrows():
        seg_rel = normpath(seg["File Location"].split(".\\")[-1])
        seg_path = (Path(args.raw_segs_folder) / seg_rel).resolve()
        seg_folder_path = str(seg_path)        # absolute, after resolve()

        ct_uid = seg["ReferencedSeriesInstanceUID"].strip()
        ct_meta = raw_meta[raw_meta["Series UID"] == ct_uid]
        if ct_meta.empty:
            continue

        ct_folder = resolve_ct_path(Path(args.raw_vols_folder),
                            ct_meta.iloc[0]["File Location"]).resolve()


        if not ct_folder.exists():
            ct_folder = Path(str(ct_folder).replace("-NA", ""))
            if not ct_folder.exists():
                print(ct_folder)
                continue

        # ---------- run *identical* preprocessing ------------------------- #
        vol, _, dcm_slices, direction = load_single_volume(str(ct_folder))
        if vol is None:
            
            continue

        if direction == "sagittal":
            vol = vol.transpose(1, 0, 2)
        elif direction == "coronal":
            vol = vol.transpose(2, 0, 1)

        occ = get_occupied_slices(str(seg_path / os.listdir(seg_path)[0]),
                                  dcm_slices, direction)
        if not occ:
            
            continue

        vol, zoom = preprocess(vol, TARGET_SHAPE)
        occ = remap_occupied_slices(occ, zoom[0])
        occ = enforce_depth(occ, vol.shape[0])      # axis-0 length
        new_arr = vol[occ]                          # shape (66, 224, 224)
        
        # ---------- compare to reference ---------------------------------- #
        def arrays_match(a, b) -> bool:
            return a.shape == b.shape and \
                   float(np.abs(a.astype(np.float32) -
                                b.astype(np.float32)).max()) <= args.epsilon

        delta = float(np.abs(new_arr.astype(np.float32) -
                             ref_arr.astype(np.float32)).max())
        if arrays_match(new_arr, ref_arr):
            logging.info(
                f"[MATCH] emb={embedding_path}  | seg={seg_folder_path} | raw={ct_folder}  |  Δ={delta:.1e}"
            )
            matches.append(
                (patient, idx, embedding_path,seg_folder_path, ct_uid, str(ct_folder), delta)
            )
            continue

        # ---------- fallback: try (224,224,66) orientation ---------------- #
        transposed = vol[:, :, occ]                 # shape (224, 224, 66)
        if arrays_match(transposed, ref_arr):
            delta2 = float(np.abs(transposed.astype(np.float32) -
                                  ref_arr.astype(np.float32)).max())
            logging.info(
                f"[MATCH] emb={embedding_path}  |  raw={ct_folder}  |  "
                f"Δ={delta2:.1e}  (transposed)"
            )
            matches.append(
                (patient, idx, embedding_path, ct_uid, str(ct_folder), delta2)
            )

    return matches


# --------------------------------------------------------------------------- #
#  Main --------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser("Match raw CT volumes to saved 224×224×66 arrays")
    ap.add_argument("--fold_file", required=True)
    ap.add_argument("--processed_folder", required=True)
    ap.add_argument("--raw_vols_folder", required=True)
    ap.add_argument("--raw_vols_metadata", required=True)
    ap.add_argument("--raw_segs_folder", required=True)
    ap.add_argument("--segs_csv", required=True)
    ap.add_argument("--segs_metadata", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--out_csv", default="matched_raw_volumes.csv")
    args = ap.parse_args(argv)

    log_file = setup_logging()
    t0 = time.time()

    # ---------- load tables ------------------------------------------------ #
    fold_df = pd.read_csv(args.fold_file)
    raw_meta_df = pd.read_csv(args.raw_vols_metadata)

    seg_report = pd.read_csv(args.segs_csv)
    seg_meta = pd.read_csv(args.segs_metadata)

    seg_df = (
        seg_report.set_index("SeriesInstanceUID")
        .join(seg_meta.set_index("Series UID")["File Location"], how="inner")
        .reset_index()
    )
    seg_df = seg_df[seg_df["Annotation Type"] == "Segmentation"]

    # ---------- create job list ------------------------------------------- #
    jobs = []
    for _, row in fold_df.iterrows():
        patient = row["patient_id"]
        m = re.search(r"(\d+)_embeddings\.npy$", row["ct_path"])
        if not m:
            logging.warning(f"Bad ct_path: {row['ct_path']}")
            continue
        idx = int(m.group(1))
        ref_np = Path(args.processed_folder) / patient / f"{idx}.npy"
        patient_segs = seg_df[seg_df["PatientID"] == patient]
        if patient_segs.empty:
            continue
        jobs.append(
            dict(
                patient_id=patient,
                idx=idx,
                embedding_path=os.path.abspath(row["ct_path"]),
                ref_np_path=str(ref_np),
                seg_df=patient_segs,
                raw_meta_df=raw_meta_df,
                args=args,
            )
        )

    logging.info(f"Fold rows queued: {len(jobs)}")

    # ---------- multiprocessing ------------------------------------------- #
    results: List[Tuple] = []
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(worker, jobs):
            results.extend(res)

    # ---------- write output ---------------------------------------------- #
    if results:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "patient_id", "idx", "embedding_path",
                "seg_folder",              # <── NEW
                "raw_series_uid", "raw_folder", "delta"
            ])
            w.writerows(results)
        logging.info(f"Wrote {len(results)} matches → {args.out_csv}")
    else:
        logging.warning("❗ No matches found.")

    logging.info(f"Elapsed: {time.time() - t0:.1f} s  |  full log → {log_file}")


if __name__ == "__main__":
    main()
