import os
import numpy as np
import pandas as pd
import concurrent.futures

import sys
sys.path.insert(0, '../util')
from data_util import *
import pandas as pd
from datetime import datetime


OPENSLIDE_PATH = "C:\\Users\\peter\/Documents\\Uni\\Second_Year\\MDP\\Openslide\\openslide-bin-4.0.0.3-windows-x64\\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
import ntpath

from multiprocessing import Pool

def thread(row):
    target_depths = {"G1":140, "G2":16, "G3":46}
    root_path = os.path.normpath('../data/raw/Dataset57PatientsCPTACPDA/manifest-1720346699071/CPTAC-PDA')
    segmentation_root = os.path.normpath('../data/raw/Segmentations/')
    DIRNAMES = 1
    subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()] #Get patients folders
    annotations = pd.read_csv('../data/clinical_annotations.tsv',  sep='\t')
    segmentations = pd.read_csv('../data/Metadata_Report_CPTAC-PDA_2023_07_14.csv')
    segmentations_metadata = pd.read_csv('../data/raw/Segmentations/metadata.csv')
    metadata = pd.read_csv('../data/raw/Dataset57PatientsCPTACPDA/manifest-1720346699071/metadata.csv')
    target_shape = [224,224,224]
    # Since the volumes are put in subfolders of patient's folders (and I'm still not sure about which of the 
    # volume subfolders I should choose) I choose the first subfolder of the patient as the volume to be 
    # preprocessed.
    segmentations = segmentations.set_index("SeriesInstanceUID").join(segmentations_metadata.set_index("Series UID")["File Location"], how = "inner")
    segmentations = segmentations[segmentations["Annotation Type"] == "Segmentation"]
    
    referenced_series_instance_uid = row["ReferencedSeriesInstanceUID"]
        
    volume_folder = metadata[metadata["Series UID"]==referenced_series_instance_uid]["File Location"]
    if volume_folder.empty:
        return None
    volume_folder = volume_folder.iloc[0]
    volume_folder = os.path.join(root_path,os.path.join(*(volume_folder.split(os.path.sep)[2:])))
    #print(volume_folder)
    vol, dim, dicom_slices = load_single_volume(volume_folder)
    #print(vol.shape)
    
    segmentation_folder = row["File Location"]
    
    #dicom.dcmread(segmentation_folder)
    seg_file = os.listdir(os.path.join(segmentation_root,segmentation_folder))[0]
    #print(os.path.join(segmentation_root,segmentation_folder,seg_file))
    occupied_slices = get_occupied_slices(os.path.join(segmentation_root,segmentation_folder,seg_file), dicom_slices)
    
    vol, zoom_factors = preprocess(vol,target_shape)
    occupied_slices = remap_occupied_slices(occupied_slices, zoom_factors[0])
    
    patient_id = row["PatientID"]
    cancer_grade = annotations.loc[annotations['case_submitter_id'] == patient_id]['tumor_grade'].iloc[0]
    
    #Oversample slices with nontumor slices if needed
    if not occupied_slices: 
        print(f"Skipped patient: {patient_id}")
        return None
    
    left_index = min(occupied_slices) - 1
    right_index = max(occupied_slices) + 1
    while len(occupied_slices) < target_depths[cancer_grade.strip()] and target_depths[cancer_grade.strip()] != "G2":
        if left_index >= 0:
            occupied_slices.insert(0, left_index)  # Add frame to the left (start of the list)
            left_index -= 1
        if len(occupied_slices) < target_depths[cancer_grade.strip()] and right_index < len(vol):
            occupied_slices.append(right_index)  # Add frame to the right (end of the list)
            right_index += 1
        
        
    
    output_path = "../data/processed_oversampling/CT/"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    i = 0
    while os.path.exists(output_path+patient_id+'/%s.npy'% i):
        i += 1
    os.makedirs(output_path+patient_id+'/', exist_ok = True)
    np.save(output_path+patient_id+'/%s.npy'% i, vol[occupied_slices])
    return patient_id, cancer_grade
    labels_f.write(patient_id+","+cancer_grade+"\n")


        
if __name__=="__main__":
    #ntpath.realpath = ntpath.abspath
    target_depths = {"G1":140, "G2":16, "G3":46}
    root_path = os.path.normpath('../data/raw/Dataset57PatientsCPTACPDA/manifest-1720346699071/CPTAC-PDA')
    segmentation_root = os.path.normpath('../data/raw/Segmentations/')
    DIRNAMES = 1
    subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()] #Get patients folders
    annotations = pd.read_csv('../data/clinical_annotations.tsv',  sep='\t')
    segmentations = pd.read_csv('../data/Metadata_Report_CPTAC-PDA_2023_07_14.csv')
    segmentations_metadata = pd.read_csv('../data/raw/Segmentations/metadata.csv')
    metadata = pd.read_csv('../data/raw/Dataset57PatientsCPTACPDA/manifest-1720346699071/metadata.csv')
    target_shape = [224,224,224]
    # Since the volumes are put in subfolders of patient's folders (and I'm still not sure about which of the 
    # volume subfolders I should choose) I choose the first subfolder of the patient as the volume to be 
    # preprocessed.
    segmentations = segmentations.set_index("SeriesInstanceUID").join(segmentations_metadata.set_index("Series UID")["File Location"], how = "inner")
    segmentations = segmentations[segmentations["Annotation Type"] == "Segmentation"]
    
    #print(segmentations.head())         
    rows = [row for index, row in segmentations.iterrows()]
    with Pool(4) as p:
        results = p.map(thread, rows)
    
    # Write the results to the labels file after processing is done
    with open("../data/processed/labels.txt", "w") as labels_f:
        for result in results:
            if result is not None:
                patient_id, cancer_grade = result
                labels_f.write(f"{patient_id},{cancer_grade}\n")
        