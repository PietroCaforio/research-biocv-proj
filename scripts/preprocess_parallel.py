import os
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
import sys
import logging
sys.path.insert(0, '../')
from util.data_util import *
from datetime import datetime
import time
import argparse
from multiprocessing import Pool
import re
if os.name=="nt":
    OPENSLIDE_PATH = "C://Users//peter//Documents//Uni//Second_Year//MDP//Openslide//openslide-bin-4.0.0.3-windows-x64//bin"
    with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
else:
    import openslide

def setup_logging():
    """Sets up the logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"processing_log_{timestamp}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logging.info("Logging setup complete.")

def change_case(str):
    # Split the string by underscores
    components = str.split('_')
    # Capitalize each component
    camel_case = ' '.join(word.capitalize() for word in components)
    camel_case = camel_case.replace("Id", "ID")
    return camel_case

def thread(params):
    #Get arguments
    args = params["args"]
    row = params["row"]
    root_path = params["root_path"]
    segmentation_root = params["segmentation_root"]
    metadata = params["metadata"]
    annotations = params["annotations"]
    validation_patients = params["validation_patients"]
    
    target_shape = [args.target_shape, args.target_shape, args.target_shape]  # Default [224,224,224]
    fix_depth = args.fix_depth 
    
    #Depths for oversampling
    target_depths = {"G1":66, "G2":66, "G3":66}
    
    #Track progress
    done_set = set(Path(f"./progress{args.dataset}.txt").read_text().splitlines())
    
    #print("Processing:",row["index"], "...")
    
    if args.progress and row["File Location"] in done_set:
        logging.info(f"{row['File Location']} already done, skipped")
        return None
        
    patient_id = row["PatientID"].strip()
    logging.info(f"Processing patient: {patient_id}")
    
    referenced_series_instance_uid = row["ReferencedSeriesInstanceUID"].strip()
    segmentation_folder = row["File Location"]    
    segmentation_folder = segmentation_folder.split('.\\')[-1]
    if os.name != "nt": segmentation_folder = segmentation_folder.replace('\\',"/").replace('-NA','')
    
    
    #Get folder location of volume to be processed associated with segmentation 
    volume_folder = metadata[metadata["Series UID"]==referenced_series_instance_uid]["File Location"]
    index = row["index"]
    if volume_folder.empty:
        logging.warning(f"Empty volume folder for patient {patient_id}. row index: {index}")
        return None
    cancer_grade = annotations.loc[annotations['Case Submitter ID'] == patient_id]['Tumor Grade'].iloc[0] #label
    
    volume_folder = volume_folder.iloc[0]
    #print(volume_folder)
    if os.name != "nt": volume_folder = volume_folder.replace('\\','/').replace('-NA','')
    volume_folder = os.path.join(root_path,os.path.join(*(volume_folder.split(os.path.sep)[2:])))
    vol, dim, dicom_slices, direction = load_single_volume(volume_folder)
    
    #Convert volume orientation to axial if needed
    if direction == "sagittal":
        vol = vol.transpose(1,0,2)
    elif direction == "coronal":
        vol = vol.transpose(2,0,1)
    else: direction = "axial"
    
    #Get segmentation path
    seg_path = os.path.join(segmentation_root,segmentation_folder)
    
    seg_path = os.path.abspath(seg_path)
    
    #Support long paths
    if os.name == "nt":
        if seg_path.startswith(u"\\\\"):
            seg_path = u"\\\\?\\UNC\\" + seg_path[2:]
        else:
            seg_path = u"\\\\?\\" + seg_path
    seg_file = os.listdir(seg_path)[0]
    
    #Get slices of volume occupied by segmentation (tumor)
    occupied_slices = get_occupied_slices(os.path.join(segmentation_root,segmentation_folder,seg_file), dicom_slices, direction)
    #Preprocess volume and convert it to target_shape
    vol, zoom_factors = preprocess(vol,target_shape)
    #Remap the segmentation slice coordinates to the new volume coordinates
    occupied_slices = remap_occupied_slices(occupied_slices, zoom_factors[0])
    #If volume has no segmentation, drop it
    if not occupied_slices:
        logging.warning(f"Skipped volume {volume_folder} for patient {patient_id}: No segmentation found.")
        return None
    
    if args.oversampling:
        #Oversampling for class imbalance (hard-coded G2 majority class)
        if patient_id not in validation_patients and cancer_grade != "G2":
            left_index = min(occupied_slices) - 1
            right_index = max(occupied_slices) + 1
            while len(occupied_slices) < target_depths[cancer_grade.strip()] and cancer_grade.strip() != "G2":
                if left_index >= 0:
                    occupied_slices.insert(0, left_index)  # Add frame to the left (start of the list)
                    left_index -= 1
                if len(occupied_slices) < target_depths[cancer_grade.strip()] and right_index < len(vol):
                    occupied_slices.append(right_index)  # Add frame to the right (end of the list)
                    right_index += 1
        elif cancer_grade == "G2": print(f"G2 patient {patient_id} not padded") 
        else: print(f"validation patient or G2 patient {patient_id} not padded")
    
    
    if args.fix_depth is not None:
        #Oversample slices with nontumor slices for padding to fix-depth
        left_index = min(occupied_slices) - 1
        right_index = max(occupied_slices) + 1
        while len(occupied_slices) < fix_depth :
            if left_index >= 0:
                occupied_slices.insert(0, left_index)  # Add frame to the left (start of the list)
                left_index -= 1
            if len(occupied_slices) < fix_depth and right_index < len(vol):
                occupied_slices.append(right_index)  # Add frame to the right (end of the list)
                right_index += 1
    output_path = args.destination
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    i = 0
    while os.path.exists(os.path.join(output_path,patient_id,'%s.npy'% i)):
        i += 1
    os.makedirs(os.path.join(output_path,patient_id), exist_ok = True)
    np.save(os.path.join(output_path,patient_id,'%s.npy'% i), vol[occupied_slices])
    with open(f"progress{args.dataset}.txt","a") as progress_file:
        progress_file.write(row["File Location"]+"\n")
    
    logging.info(f"Successfully processed patient {patient_id}")
    
    return patient_id, cancer_grade


def main(args):
    setup_logging()
    # Given the dataset selected load the needed files for preprocessing (segmentations, annotations, metadata ...)
    if args.dataset == "CPTAC_UCEC":
        segmentations = pd.read_csv("../data/Metadata_Report_CPTAC-UCEC_2023_07_14.csv")
        segmentations_metadata = pd.read_csv('../data/raw/69PatientsCPTACUCEC/manifest-1728901427271/metadata.csv')
        segmentations = segmentations.set_index("SeriesInstanceUID").join(segmentations_metadata.set_index("Series UID")["File Location"], how = "inner")
        segmentations = segmentations[segmentations["Annotation Type"] == "Segmentation"]

        clinical_data_list = [
            pd.read_csv("../data/CPTACUCEC_clinicalannotationsProteom.csv"),
            pd.read_csv("../data/CPTACUCEC_clinicalIDC.csv"),
            pd.read_csv("../data/CPTACUCEC_clinicalConfirmatoryGlyco.csv"),
            pd.read_csv("../data/CPTACUCEC_.clinicalConfirmatoryProteome.csv"),
            pd.read_csv("../data/CPTACUCEC_clinicalannotations.csv"),
            pd.read_csv("../data/CPTACUCEC_clinicalannotationsAcetylome.csv"),
            pd.read_csv("../data/CPTACUCEC_clinicalannotationsPhosphoproteom.csv")
        ]
        annotations = pd.concat(clinical_data_list)
        root_path = os.path.normpath('../data/raw/69PatientsCPTACUCEC/manifest-1728901427271/CPTAC-UCEC')
        segmentation_root = os.path.normpath('../data/raw/69PatientsCPTACUCEC/manifest-1728901427271/')
        metadata = pd.read_csv('../data/raw/69PatientsCPTACUCEC/manifest-1728901427271/metadata.csv')
        validation_patients = Path("../data/processed_CPTACUCEC_3D/val.txt").read_text().splitlines()    
    elif args.dataset == "CPTAC_PDA":
        
        segmentations = pd.read_csv('../data/Metadata_Report_CPTAC-PDA_2023_07_14.csv')
        segmentations_metadata = pd.read_csv('../data/raw/Segmentations/metadata.csv')
        segmentations = segmentations.set_index("SeriesInstanceUID").join(segmentations_metadata.set_index("Series UID")["File Location"], how = "inner")
        segmentations = segmentations[segmentations["Annotation Type"] == "Segmentation"]

        root_path = os.path.normpath('../data/raw/Dataset57PatientsCPTACPDA/manifest-1720346699071/CPTAC-PDA')
        segmentation_root = os.path.normpath('../data/raw/Segmentations/')
        metadata = pd.read_csv('../data/raw/Dataset57PatientsCPTACPDA/manifest-1720346699071/metadata.csv')    
        annotations = pd.read_csv('../data/clinical_annotations.tsv',  sep='\t')
        annotations.columns = annotations.columns.to_series().apply(change_case)
        validation_patients = Path("../data/processed_oversampling/val.txt").read_text().splitlines()
    else: 
        logging.error(f"{args.dataset} is not a valid option. Please choose either CPTAC_PDA or CPTAC_UCEC.")
        return
    
    segmentations.index.name = 'index'
    
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
            "validation_patients": validation_patients.copy()
        }
        for index, row in segmentations.iterrows()
    ]
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
    with open(os.path.join('/'.join(args.destination.split('/')[:-2]),"labels.txt"), "w") as labels_f:
        for result in results:
            if result is not None:
                patient_id, cancer_grade = result
                labels_f.write(f"{patient_id},{cancer_grade}\n")
                
    logging.info("Processing complete.")

if __name__=="__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--oversampling", type = bool, default=False)
    parser.add_argument("--destination", type=str, default="../data/processed/CT/")
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--fix_depth", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="CPTAC_PDA")
    parser.add_argument("--progress", type = bool, default=False) 
    parser.add_argument("--target_shape", type = int, default=224) 
    parser.add_argument("--debug", type = bool, default=False) 
    
    args = parser.parse_args()
    main(args)
    end = time.time()
    logging.info(f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")   
