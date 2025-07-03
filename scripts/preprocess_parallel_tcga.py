import argparse
import logging
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "../")  # noqa: E402

from util.data_util import load_single_volume  # noqa: E402
from util.data_util import preprocess  # noqa: E402
from util.data_util import remap_occupied_slices  # noqa: E402

import os, math, logging
import numpy as np
import pydicom

def get_occupied_slices(seg_path, dicom_slices, direction="axial",
                        keep_segments=None,  # e.g. {1,2,3}
                        z_tol=1.0):
    """
    Works for *both* RT STRUCT and DICOM-SEG.
    ------------------------------------------------------------------
    seg_path        -> absolute path to the .dcm file (SEG or RTSTRUCT)
    dicom_slices    -> list returned by load_single_volume()
    direction       -> "axial" | "coronal" | "sagittal"
    keep_segments   -> optional {segment numbers} to keep
    z_tol           -> z-match tolerance in mm
    Returns: sorted list of occupied slice indices
    """
    seg_ds = pydicom.dcmread(seg_path, force=True)  # works for both SOP classes

    # ------------------------------------------------------------
    # 1. Build CT slice z-coordinate vector once
    if direction == "sagittal":
        slice_z = np.array([float(s.ImagePositionPatient[0]) for s in dicom_slices])
    elif direction == "coronal":
        slice_z = np.array([float(s.ImagePositionPatient[1]) for s in dicom_slices])
    else:
        slice_z = np.array([float(s.ImagePositionPatient[2]) for s in dicom_slices])

    occupied = set()

    # ------------------------------------------------------------
    # 2. Branch by modality
    modality = seg_ds.Modality.upper()

    if modality == "SEG":
        # a) pixel_array → shape (n_frames, rows, cols)
        masks = seg_ds.pixel_array  # pydicom handles unpacking of 1-bit data

        # b) loop over frames
        fgs = seg_ds.PerFrameFunctionalGroupsSequence
        for i, fg in enumerate(fgs):
            # Segment number of this frame
            seg_number = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
            if keep_segments and seg_number not in keep_segments:
                continue

            # Skip empty frames quickly
            if not masks[i].any():
                continue

            # c) z-coordinate of this frame
            ipp = fg.PlanePositionSequence[0].ImagePositionPatient
            if direction == "sagittal":
                z_val = float(ipp[0])
            elif direction == "coronal":
                z_val = float(ipp[1])
            else:
                z_val = float(ipp[2])

            # d) find matching CT slice
            idx = np.where(np.isclose(slice_z, z_val, atol=z_tol))[0]
            if idx.size:
                occupied.add(int(idx[0]))

    elif modality in ("RTSTRUCT", "RT STRUCT"):
        # ← old behaviour, unchanged except we use the same variables
        for roi in seg_ds.ROIContourSequence:
            for contour in roi.ContourSequence:
                pts = contour.ContourData
                if direction == "axial":
                    z_vals = pts[2::3]
                elif direction == "coronal":
                    z_vals = pts[1::3]
                else:  # sagittal
                    z_vals = pts[0::3]

                for z in z_vals:
                    idx = np.where(np.isclose(slice_z, z, atol=z_tol))[0]
                    if idx.size:
                        occupied.add(int(idx[0]))
    else:
        logging.error(f"Unsupported segmentation modality {modality}")
        return []

    return sorted(occupied)

def setup_logging():
    """Sets up the logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/processing_log_{timestamp}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logging.info("Logging setup complete.")


def change_case(str):
    # Split the string by underscores
    components = str.split("_")
    if len(components) <= 1:
        return str
    # Capitalize each component
    camel_case = " ".join(word.capitalize() for word in components)
    camel_case = camel_case.replace("Id", "ID")
    camel_case = camel_case.replace("id", "ID")
    return camel_case


def thread(params):
    # Get arguments
    args = params["args"]
    row = params["row"]
    root_path = params["root_path"]
    segmentation_root = params["segmentation_root"]
    metadata = params["metadata"]
    annotations = params["annotations"]
    validation_patients = params["validation_patients"]

    target_shape = [
        args.target_shape,
        args.target_shape,
        args.target_shape,
    ]  # Default [224,224,224]
    fix_depth = args.fix_depth

    # Depths for oversampling
    target_depths = {"G1": 66, "G2": 66, "G3": 66}

    # Track progress
    done_set = set(
        Path(f"./progress/progress{args.dataset}.txt").read_text().splitlines()
    )

    # print("Processing:",row["index"], "...")

    if args.progress and row["File Location"] in done_set:
        logging.info(f"{row['File Location']} already done, skipped")
        return None

    patient_id = row["PatientID"].strip()
    logging.info(f"Processing patient: {patient_id}")

    referenced_series_instance_uid = row["ReferencedSeriesInstanceUID"].strip()
    segmentation_folder = row["File Location"]
    segmentation_folder = segmentation_folder.split(".\\")[-1]
    if os.name != "nt":
        segmentation_folder = segmentation_folder.replace("\\", "/")
        # Get segmentation path
        seg_path = os.path.join(segmentation_root, segmentation_folder)

        seg_path = os.path.abspath(seg_path)
        if not os.path.exists(seg_path):
            segmentation_folder = segmentation_folder.replace("-NA", "")

    seg_path = os.path.join(segmentation_root, segmentation_folder)
    seg_path = os.path.abspath(seg_path)
    # Support long paths
    if os.name == "nt":
        if seg_path.startswith("\\\\"):
            seg_path = "\\\\?\\UNC\\" + seg_path[2:]
        else:
            seg_path = "\\\\?\\" + seg_path

    #seg_file = os.listdir(seg_path)[0]

    # Get folder location of volume to be processed associated with segmentation
    volume_folder = metadata[metadata["Series UID"] == referenced_series_instance_uid][
        "File Location"
    ]
    index = row["index"]
    if volume_folder.empty:
        logging.warning(
            f"Empty volume folder for patient {patient_id}. row index: {index}."
        )
        return None

    cancer_grade_df = annotations.loc[annotations["Case Submitter ID"] == patient_id][
        "Tumor Grade"
    ]
    # if cancer_grade_df.empty:
    #    logging.warning(f"No annotation available for patient {patient_id}. Skipped.")
    #    return None

    # cancer_grade = cancer_grade_df.iloc[0]  # label
    cancer_grade = None
    volume_folder = volume_folder.iloc[0]
    # print(volume_folder)

    if os.name != "nt":
        volume_folder = volume_folder.replace("\\", "/")
        volume_path = os.path.join(
            root_path, os.path.join(*(volume_folder.split(os.path.sep)[2:]))
        )
        if not os.path.exists(volume_path):
            volume_folder = volume_folder.replace("-NA", "")
            volume_path = os.path.join(
                root_path, os.path.join(*(volume_folder.split(os.path.sep)[2:]))
            )
        volume_folder = volume_path
    vol, dim, dicom_slices, direction = load_single_volume(volume_folder)
    if vol is None:
        logging.warning(f"Empty volume, no ImageOrientation for {volume_folder}")
        return None
    # Convert volume orientation to axial if needed
    if direction == "sagittal":
        vol = vol.transpose(1, 0, 2)
    elif direction == "coronal":
        vol = vol.transpose(2, 0, 1)
    else:
        direction = "axial"

    # Get slices of volume occupied by segmentation (tumor)
    occupied_slices = get_occupied_slices(
        os.path.join(segmentation_root, seg_path),
        dicom_slices,
        direction,
        keep_segments={2}
    )
    # Preprocess volume and convert it to target_shape
    vol, zoom_factors = preprocess(vol, target_shape)
    # Remap the segmentation slice coordinates to the new volume coordinates
    occupied_slices = remap_occupied_slices(occupied_slices, zoom_factors[0])
    # If volume has no segmentation, drop it
    if not occupied_slices:
        logging.warning(
            f"Skipped volume {volume_folder} for patient {patient_id}: No segmentation found."
        )
        return None

    if args.oversampling:
        # Oversampling for class imbalance (hard-coded G2 majority class)
        if patient_id not in validation_patients and cancer_grade != "G2":
            left_index = min(occupied_slices) - 1
            right_index = max(occupied_slices) + 1
            while (
                len(occupied_slices) < target_depths[cancer_grade.strip()]
                and cancer_grade.strip() != "G2"
            ):
                if left_index >= 0:
                    occupied_slices.insert(
                        0, left_index
                    )  # Add frame to the left (start of the list)
                    left_index -= 1
                if len(occupied_slices) < target_depths[
                    cancer_grade.strip()
                ] and right_index < len(vol):
                    occupied_slices.append(
                        right_index
                    )  # Add frame to the right (end of the list)
                    right_index += 1
        elif cancer_grade == "G2":
            print(f"G2 patient {patient_id} not padded")
        else:
            print(f"validation patient or G2 patient {patient_id} not padded")

    if args.fix_depth is not None:
        # Oversample slices with nontumor slices for padding to fix-depth
        left_index = min(occupied_slices) - 1
        right_index = max(occupied_slices) + 1
        while len(occupied_slices) < fix_depth:
            if left_index >= 0:
                occupied_slices.insert(
                    0, left_index
                )  # Add frame to the left (start of the list)
                left_index -= 1
            if len(occupied_slices) < fix_depth and right_index < len(vol):
                occupied_slices.append(
                    right_index
                )  # Add frame to the right (end of the list)
                right_index += 1
    output_path = args.destination

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    i = 0
    while os.path.exists(os.path.join(output_path, patient_id, "%s.npy" % i)):
        i += 1
    os.makedirs(os.path.join(output_path, patient_id), exist_ok=True)
    np.save(os.path.join(output_path, patient_id, "%s.npy" % i), vol[occupied_slices])
    with open(f"./progress/progress{args.dataset}.txt", "a") as progress_file:
        progress_file.write(row["File Location"] + "\n")

    logging.info(f"Successfully processed patient {patient_id}")

    return patient_id, cancer_grade


def main(args):
    setup_logging()

    if args.load_args:
        args_df = pd.read_csv(args.load_args, quotechar='"')

        for key in args_df.columns:
            key = key.strip()
            setattr(args, str(key), str(args_df.loc[0, key]))

    segmentations = pd.read_csv(args.segmentations)
    segmentations_metadata = pd.read_csv(args.segmentations_metadata)
    segmentations = segmentations.set_index("SeriesInstanceUID").join(
        segmentations_metadata.set_index("Series UID")["File Location"], how="inner"
    )
    segmentations = segmentations[segmentations["Annotation Type"] == "Segmentation"]

    clinical_data_list = [
        pd.read_csv(path) if path.endswith(".csv") else pd.read_csv(path, sep="\t")
        for path in args.clinical_data_list.split(";")
    ]
    annotations = pd.concat(clinical_data_list)

    root_path = os.path.normpath(args.root_path)
    segmentation_root = os.path.normpath(args.segmentation_root)
    metadata = pd.read_csv(args.metadata)
    
    annotations.columns = annotations.columns.to_series().apply(change_case)
    validation_patients = Path(args.validation_patients).read_text().splitlines()

    segmentations.index.name = "index"

    segmentations = segmentations.reset_index()
    # Prepare input for thread pool
    rows = [
        {
            "row": row,
            "args": args,
            "annotations": annotations,
            "root_path": root_path,
            "segmentation_root": segmentation_root,
            "metadata": metadata.copy(),
            "validation_patients": validation_patients.copy(),
        }
        for index, row in segmentations.iterrows()
    ]

    print(rows)
    if args.debug:
        results = []
        for row in rows:
            results.extend(thread(row))
    else:
        with Pool(args.np) as p:
            results = p.map(thread, rows)

    # Each thread returns PatientID, cancer_grade pairs
    results = set(results)

    # Generate the labels file
    with open(
        os.path.join("/".join(args.destination.split("/")[:-2]), "labels.txt"), "w"
    ) as labels_f:
        for result in results:
            if result is not None:
                patient_id, cancer_grade = result
                labels_f.write(f"{patient_id},{cancer_grade}\n")

    logging.info("Processing complete.")


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--oversampling", type=bool, default=False)
    parser.add_argument("--destination", type=str, default="../data/processed/CT/")
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--fix_depth", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="CPTAC_PDA")
    parser.add_argument("--progress", type=bool, default=False)
    parser.add_argument("--target_shape", type=int, default=224)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument(
        "--load_args",
        type=str,
        default=None,
        help="Select (.csv) from which you can load arguments",
    )

    parser.add_argument(
        "--segmentations", type=str, help="Segmentations report (.csv) metadata"
    )
    parser.add_argument(
        "--segmentations_metadata",
        type=str,
        help="Segmentation's nbia data retriever (.csv) metadata",
    )
    parser.add_argument(
        "--clinical_data_list",
        type=str,
        help="List of clinical metadata (.csv) files. Single str if from shell",
    )
    parser.add_argument("--root_path", type=str, help="Volumes's root path")
    parser.add_argument(
        "--segmentation_root", type=str, help="Segmentation's dcm folder"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Volumes's metadata (.csv) file from nbia-data-retriever",
    )
    parser.add_argument(
        "--validation_patients", type=str, help="List of validation's patients (.txt)"
    )

    args = parser.parse_args()
    main(args)
    end = time.time()
    logging.info(f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")
