import pandas as pd
import os

clinical_data_list = [
    pd.read_csv("./data/metadata_annotations/CPTACPDAclinicalSerumGlycoproteome.csv"), 
    pd.read_csv("./data/metadata_annotations/clinical_annotations.tsv", sep="\t").rename(columns={"tumor_grade":"Tumor Grade","case_submitter_id":"Case Submitter ID"}), 
    pd.read_csv("./data/metadata_annotations/CPTAC2clinical.tsv", sep="\t").rename(columns={"tumor_grade":"Tumor Grade","case_submitter_id":"Case Submitter ID"}),
    pd.read_csv("./data/metadata_annotations/CPTACPDAclinical.csv"), 
    pd.read_csv("./data/metadata_annotations/CPTACPDAclinicalDIAProteome.csv"), 
    pd.read_csv("./data/metadata_annotations/CPTACPDAclinicalBioTextProteome.csv"), 
    pd.read_csv("./data/metadata_annotations/CPTACPDAclinicalSerumProteome.csv")
    ]

#clinical_data = pd.read_csv("./data/CPTACUCEC_clinicalannotationsProteom.csv")
#clinical_data1 = pd.read_csv("./data/CPTACUCEC_clinicalIDC.csv")

radiology_patients = pd.read_csv("data/metadata_annotations/Metadata_Report_CPTAC-PDA_2023_07_14.csv")
radiology_patients = radiology_patients[radiology_patients["Annotation Type"]=="Segmentation"]["PatientID"]
radiology_patients_list = radiology_patients.to_list()

#print(radiology_patients)
labeled_patients = []
for clinical_data in clinical_data_list:
    labeled_patients.extend(clinical_data.loc[clinical_data["Tumor Grade"] == "G1"]["Case Submitter ID"].to_list()) 
#print(g1_patients)


#g1_annotated = set(g1_patients) & set(radiology_patients) #intersection between reference patients and g1_patients
#g1_volumes = radiology_patients.loc[radiology_patients.isin(g1_patients)]
#print(f"g1:{len(g1_annotated)}")
#print(f"g1 volumes: {len(g1_volumes)}")
g2_patients = []
for clinical_data in clinical_data_list:
    g2_patients.extend(clinical_data.loc[clinical_data["Tumor Grade"] == "G2"]["Case Submitter ID"].to_list()) 
#print(g2_patients)
#g2_annotated = set(g2_patients) & set(radiology_patients) #intersection between reference patients and g1_patients
#g2_volumes = radiology_patients.loc[radiology_patients.isin(g2_patients)]
#print(f"g2:{len(g2_annotated)}")
#print(f"g2 volumes: {len(g2_volumes)}")

g3_patients = []
for clinical_data in clinical_data_list:
    g3_patients.extend(clinical_data.loc[clinical_data["Tumor Grade"] == "G3"]["Case Submitter ID"].to_list()) 
#print(g3_patients)
#g3_volumes = radiology_patients.loc[radiology_patients.isin(g3_patients)]
#g3_annotated = set(g3_patients) & set(radiology_patients) #intersection between reference patients and g1_patients
#print(f"g3:{len(g3_annotated)}")
#print(f"g2 volumes: {len(g3_volumes)}")
labeled_patients.extend(g2_patients)
labeled_patients.extend(g3_patients)
print(len(set(labeled_patients)))

# Get all files in the folder
folder_files = os.listdir("../CLAM/CPTAC_PDA")

# Extract base IDs from folder files (removing the -XX.svs suffix)
histology_patients = set()
for filename in folder_files:
    if filename.endswith('.svs'):
        # Split at the last hyphen and take the first part
        base_id = filename.rsplit('-', 1)[0]
        histology_patients.add(base_id)
histology_patients = histology_patients & set(labeled_patients)
print("pazienti per cui ho l'istologia con label",len(histology_patients))
print("Pazienti per cui ho istologia ma non radiologia",len(set(histology_patients) & (set(labeled_patients) - set(radiology_patients)))) # Per quanti pazienti ho histo ma non radiologia
#print(len(set(g1_patients) - set(radiology_patients)))
print("Pazienti per cui ho radiologia ma non l'istologia",len( (set(labeled_patients) & set(radiology_patients)) - set(histology_patients) ))

#Calculate list of labeled patients for which I have either histology or radiology 

full_patients = histology_patients.union( set(radiology_patients) & set(labeled_patients) )


print(len(full_patients))
for patient in full_patients:
    for clinical_data in clinical_data_list:
        tumor_grade = clinical_data[clinical_data["Case Submitter ID"] == patient]["Tumor Grade"]
        if not tumor_grade.empty and tumor_grade.to_list()[0] != "Not Reported":
            break
    print(patient+",", tumor_grade.to_list()[0])
