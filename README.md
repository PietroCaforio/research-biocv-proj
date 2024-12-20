# Multidisciplinary Project: Multimodality Biomedical Computervision

## Preprocessed Datasets

* [Preprocessed57PatientsCPTAC-PDA](https://drive.google.com/file/d/1vE8PcgubAyb7EzWB2-L_iqCMeCApj3u0/view?usp=drive_link)
* [Preprocessed57PatientsCPTAC-PDA (Kaggle)](https://www.kaggle.com/datasets/pietrocaforio/preprocessed57patientscptacpda/)

## Setup

### Environment Setup

1. Conda is recommended.
2. In the project folder, run `conda env create -f conda_requirements.yml` to create the environment.
3. If running on Windows, update the `OPENSLIDE_PATH` in `util/data_util.py` and `scripts/preprocess_parallel.py` to your local Windows openslide binary path.

### Download Raw Datasets

1. Download the `nbia-data-retriever` from [NBIA Data Retriever Wiki](https://wiki.cancerimagingarchive.net/display/NBIA/Installing+the+NBIA+Data+Retriever).

2. CPTAC-PDA:
   * `cd data/raw/Dataset57PatientsCPTACPDA`
   * `nbia-data-retriever --cli manifest-1720346699071.tcia -d ./`
   * `cd data/raw`
   * `nbia-data-retriever --cli Segmentations.tcia -d ./`

3. CPTAC-UCEC:
   * `cd data/raw/69PatientsCPTACUCEC`
   * `nbia-data-retriever --cli manifest-1728901427271.tcia -d ./`

## Preprocessing

1. CPTAC-PDA:
   * `cd scripts`
   * `./run_preprocess_CPTAC_PDA3D.sh`

2. CPTAC-UCEC:
   * `cd scripts`
   * `./run_preprocess_CPTAC_UCEC3D.sh`