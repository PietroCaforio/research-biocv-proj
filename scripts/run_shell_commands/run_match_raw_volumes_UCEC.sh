python scripts/match_raw_volumes.py \
  --fold_file ./interpretability/interpretability_UCEC_mixed30/fold_1_log.csv \
  --processed_folder ./data/processed/processed_CPTACUCEC_survival/MRI \
  --raw_vols_folder ./data/raw/CPTACUCEC34SurvMRI/manifest/CPTAC-UCEC \
  --raw_vols_metadata ./data/raw/CPTACUCEC34SurvMRI/manifest/metadata.csv \
  --raw_segs_folder ./data/raw/CPTACUCEC34SurvMRI/manifest \
  --segs_csv ./data/metadata_annotations/Metadata_Report_CPTAC-UCEC_2023_07_14.csv \
  --segs_metadata ./data/raw/CPTACUCEC34SurvMRI/manifest/metadata.csv \
  --workers 6 \
  --epsilon 1e-6
