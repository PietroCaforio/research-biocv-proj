import os
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
import sys
sys.path.insert(0, '../')
from util.data_util import *
import pandas as pd
from datetime import datetime
import time
import argparse

OPENSLIDE_PATH = "C://Users//peter//Documents//Uni//Second_Year//MDP//Openslide//openslide-bin-4.0.0.3-windows-x64//bin"
with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
import ntpath

from multiprocessing import Pool

def thread_CPTACUCEC(params):
    args = params["args"]
    row = params["row"]
    target_depths = {"G1":66, "G2":66, "G3":66}
    fix_depth = args.fix_depth 
    root_path = os.path.normpath('../data/raw/69PatientsCPTACUCEC/manifest-1728901427271/CPTAC-UCEC')
    segmentation_root = os.path.normpath('../data/raw/69PatientsCPTACUCEC/manifest-1728901427271/')
    metadata = pd.read_csv('../data/raw/69PatientsCPTACUCEC/manifest-1728901427271/metadata.csv')
    target_shape = [350,350,350]
    #annotations = pd.read_csv('../data/clinical_annotations.tsv',  sep='\t')
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
    
    validation_patients = Path("../data/processed_CPTACUCEC_3D/val.txt").read_text().splitlines()
    referenced_series_instance_uid = row["ReferencedSeriesInstanceUID"]
    
    volume_folder = metadata[metadata["Series UID"]==referenced_series_instance_uid]["File Location"]
    if volume_folder.empty:
        return None
    patient_id = row["PatientID"].strip()
    if args.skip_folders and patient_id in os.listdir(args.destination):
        return None
    volume_folder = volume_folder.iloc[0]
    volume_folder = os.path.join(root_path,os.path.join(*(volume_folder.split(os.path.sep)[2:])))
    #print(volume_folder)
    vol, dim, dicom_slices, direction = load_single_volume(volume_folder)
    #print(vol.shape)
    if direction == "sagittal":
        vol = vol.transpose(1,0,2)
    elif direction == "coronal":
        vol = vol.transpose(2,0,1)
    
        
    segmentation_folder = row["File Location"]
    
    #dicom.dcmread(segmentation_folder)
    segmentation_folder = segmentation_folder.split('.\\')[-1]
    seg_path = os.path.join(segmentation_root,segmentation_folder)
    seg_path = os.path.abspath(seg_path)
    if seg_path.startswith(u"\\\\"):
        seg_path = u"\\\\?\\UNC\\" + seg_path[2:]
    else:
        seg_path = u"\\\\?\\" + seg_path
    seg_file = os.listdir(seg_path)[0]
    #print(os.path.join(segmentation_root,segmentation_folder,seg_file))
    occupied_slices = get_occupied_slices(os.path.join(segmentation_root,segmentation_folder,seg_file), dicom_slices, direction)
    
    
    
    vol, zoom_factors = preprocess(vol,target_shape)
    occupied_slices = remap_occupied_slices(occupied_slices, zoom_factors[0])
    
    cancer_grade = annotations.loc[annotations['Case Submitter ID'] == patient_id]['Tumor Grade'].iloc[0]
    
    #Oversample slices with nontumor slices if needed
    if not occupied_slices: 
        print(f"Skipped patient: {patient_id}")
        return None
    if args.oversampling:
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
    return patient_id, cancer_grade

def thread_CPTACPDA(params):
    args = params["args"]
    row = params["row"]
    target_depths = {"G1":66, "G2":66, "G3":66}
    fix_depth = args.fix_depth 
    root_path = os.path.normpath('../data/raw/Dataset57PatientsCPTACPDA/manifest-1720346699071/CPTAC-PDA')
    segmentation_root = os.path.normpath('../data/raw/Segmentations/')
    metadata = pd.read_csv('../data/raw/Dataset57PatientsCPTACPDA/manifest-1720346699071/metadata.csv')
    target_shape = [224,224,224]
    annotations = pd.read_csv('../data/clinical_annotations.tsv',  sep='\t')
    validation_patients = Path("../data/processed_oversampling/val.txt").read_text().splitlines()
    referenced_series_instance_uid = row["ReferencedSeriesInstanceUID"]
    
    volume_folder = metadata[metadata["Series UID"]==referenced_series_instance_uid]["File Location"]
    if volume_folder.empty:
        return None
    patient_id = row["PatientID"].strip()
    if args.skip_folders and patient_id in os.listdir(os.path.join(args.destination,"/CT")):
        return None
    volume_folder = volume_folder.iloc[0]
    volume_folder = os.path.join(root_path,os.path.join(*(volume_folder.split(os.path.sep)[2:])))
    #print(volume_folder)
    vol, dim, dicom_slices, direction = load_single_volume(volume_folder)
    #print(vol.shape)
    
    segmentation_folder = row["File Location"]
    
    #dicom.dcmread(segmentation_folder)
    seg_file = os.listdir(os.path.join(segmentation_root,segmentation_folder))[0]
    #print(os.path.join(segmentation_root,segmentation_folder,seg_file))
    occupied_slices = get_occupied_slices(os.path.join(segmentation_root,segmentation_folder,seg_file), dicom_slices)
    
    
    
    vol, zoom_factors = preprocess(vol,target_shape)
    occupied_slices = remap_occupied_slices(occupied_slices, zoom_factors[0])
    
    cancer_grade = annotations.loc[annotations['case_submitter_id'] == patient_id]['tumor_grade'].iloc[0]
    
    #Oversample slices with nontumor slices if needed
    if not occupied_slices: 
        print(f"Skipped patient: {patient_id}")
        return None
    if args.oversampling:
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
    while os.path.exists(output_path+patient_id+'/%s.npy'% i):
        i += 1
    os.makedirs(output_path+patient_id+'/', exist_ok = True)
    np.save(output_path+patient_id+'/%s.npy'% i, vol[occupied_slices])
    return patient_id, cancer_grade
    

        
if __name__=="__main__":
    #Save timestamp
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_folders", type = bool, default=False) 
    parser.add_argument("--oversampling", type = bool, default=False)
    parser.add_argument("--destination", type=str, default="../data/processed/CT/")
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--fix_depth", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="CPTAC_PDA")
    
    args = parser.parse_args()
    
    if args.dataset == "CPTAC_PDA":
        segmentations = pd.read_csv('../data/Metadata_Report_CPTAC-PDA_2023_07_14.csv')
        segmentations_metadata = pd.read_csv('../data/raw/Segmentations/metadata.csv')
        segmentations = segmentations.set_index("SeriesInstanceUID").join(segmentations_metadata.set_index("Series UID")["File Location"], how = "inner")
        segmentations = segmentations[segmentations["Annotation Type"] == "Segmentation"]
    else:
        segmentations = pd.read_csv("../data/Metadata_Report_CPTAC-UCEC_2023_07_14.csv")
        segmentations_metadata = pd.read_csv('../data/raw/69PatientsCPTACUCEC/manifest-1728901427271/metadata.csv')
        segmentations = segmentations.set_index("SeriesInstanceUID").join(segmentations_metadata.set_index("Series UID")["File Location"], how = "inner")
        segmentations = segmentations[segmentations["Annotation Type"] == "Segmentation"]
        
    #print(segmentations.head())         
    rows = [{"row":row, "args":args} for index, row in segmentations.iterrows()]
    if args.dataset == "CPTAC_PDA":
        with Pool(args.np) as p:
            results = p.map(thread_CPTACPDA, rows)
    elif args.dataset == "CPTAC_UCEC":
        with Pool(args.np) as p:
            results = p.map(thread_CPTACUCEC, rows)
    
    results = set(results)
    # Write the results to the labels file after processing is done
    with open(os.path.join('/'.join(args.destination.split('/')[:-1]),"labels.txt"), "w") as labels_f:
        for result in results:
            if result is not None:
                patient_id, cancer_grade = result
                labels_f.write(f"{patient_id},{cancer_grade}\n")
    end = time.time()
    print("time elapsed: ",time.strftime('%H:%M:%S',time.gmtime(end - start)))    