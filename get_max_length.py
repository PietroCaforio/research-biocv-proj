import os
import numpy as np
root_path = "./data/processed_CPTACUCEC_3D_HR_PAD/CT"
subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()] #Get patients folders
empty_count = 0
empty_patients = []
max_length = 0
for subfolder in subfolders:
    patient_id = subfolder.split('/')[-1]
    volumes = [f.path for f in os.scandir(subfolder) ]
    for volume in volumes:
        length = len(np.load(volume))
        if length == 0:
            print(f"!Warning! empty volume! patient_id:{patient_id}")
        #print(f"{patient_id}, volume length: {length}")
        if length != 214:
            print(f"length: {length}\n patient_id:{patient_id}") 
        if  length > max_length:
            max_length = length
print(max_length)
