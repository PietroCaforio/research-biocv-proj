python scripts/match_raw_volumes.py \
  --fold_file ./interpretability/interpretability_PDA_mixed5/fold_2_log.csv \
  --processed_folder ./data/processed/processed_CPTAC_PDA_survival/CT \
  --raw_vols_folder ./data/raw/CPTAC_PDA_93_surv/cptacpda_93/CPTAC-PDA \
  --raw_vols_metadata ./data/raw/CPTAC_PDA_93_surv/cptacpda_93/metadata.csv \
  --raw_segs_folder ./data/raw/CPTAC_PDA_93_surv/Segmentations/ \
  --segs_csv ./data/metadata_annotations/Metadata_Report_CPTAC-PDA_2023_07_14.csv \
  --segs_metadata ./data/raw/CPTAC_PDA_93_surv/Segmentations/metadata.csv \
  --workers 6 \
  --epsilon 1e-6
