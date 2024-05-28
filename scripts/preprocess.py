import sys
sys.path.insert(0, '../util')
from util.data_util import *
import pandas as pd

# TODO: Refactor the following code in order for it to be usable as a command line script


root_path = os.path.normpath('./data/raw/manifest-1716725357109/CPTAC-PDA')
DIRNAMES = 1
subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()] #Get patients folders
target_shape = [16, 256, 256]
annotations = pd.read_csv('./data/clinical_annotations.tsv', sep='\t')
# Since the volumes are put in subfolders of patient's folders (and I'm still not sure about which of the 
# volume subfolders I should choose) I choose the first subfolder of the patient as the volume to be 
# preprocessed. 
for subfolder in subfolders:
    first_level_subfolders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
    first_subfolder = first_level_subfolders[0] #Get first subfolder
    second_level_subfolders = [f.path for f in os.scandir(first_subfolder) if f.is_dir()]
    second_subfolder = second_level_subfolders[0] #Get first sub sub folder
    print(subfolder)
    vol, dim = load_single_volume(second_subfolder)
    #Preprocessing of vol
    vol = preprocess(vol, target_shape)
    #Save vol in correct subdir based on cancer grade annotation
    patient_id = subfolder.split("\\")[-1]
    cancer_grade = annotations.loc[annotations['case_submitter_id'] == patient_id]['tumor_grade'].iloc[0]
    print(cancer_grade)
    np.save("./data/processed/CT/"+cancer_grade+"/"+patient_id,vol)


dataset_path = os.path.normpath('../../Dataset5PatientsCPTAC/WSI/')
DIRNAMES = 1
annotations = pd.read_csv('clinical_annotations.tsv', sep='\t')
wsi_ext = '.svs'
patch_size = [224,224]
threshold = 0.9 #tissue percentage per patch
max_patches = 16 #The number of patches we want per patient
for path, _, files in sorted(os.walk(dataset_path)): 
      for filename in (sorted(files)): 
          if filename.endswith (wsi_ext):
            patient_id = filename.split('.')[0]
            patient_id = patient_id[:-3]
            print(patient_id)
            slide = read_wsi(os.path.join(dataset_path,filename))
            best_level, best_patches = find_best_level(slide, patch_size, threshold, max_patches)
            
            cancer_grade = annotations.loc[annotations['case_submitter_id'] == patient_id]['tumor_grade'].iloc[0]
            output_path = './preprocessed/WSI/'+cancer_grade+'/'+patient_id+'/'
            save_patches_as_numpy(best_patches, output_path, patient_id)


