import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------------- #
#                              helpers                                       #
# ------------------------------------------------------------------------- #
class VolumeStandardizer:
    """Utility for depth-standardising 3-D CT volumes."""

    def __init__(self, method: str = "padding"):
        assert method in ["padding"], f"Unknown std. method: {method}"
        self.method = method

    def __call__(self, volume: np.ndarray, target_depth: int) -> np.ndarray:
        if self.method == "padding":
            return self._pad_depth(volume, target_depth)

    @staticmethod
    def _pad_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
        d, h, w = volume.shape
        if d >= target_depth:
            return volume
        before = (target_depth - d) // 2
        after = target_depth - d - before
        return np.pad(volume,
                      ((before, after), (0, 0), (0, 0)),
                      mode="constant",
                      constant_values=0)


# ------------------------------------------------------------------------- #
#                              dataset                                       #
# ------------------------------------------------------------------------- #
class UnimodalCTDatasetSurv(Dataset):
    """
    Unimodal survival dataset for *CT only*.

    * Identical split / fold handling to `MultimodalCTWSIDatasetSurv`
      – controlled by `labels_splits_path`, `fold`, `split`.
    * Loads every `.npy` CT file it finds in `<ct_path>/<case_id>/`.
    * Optionally depth-standardises all scans to the **maximum** depth in
      the split (same idea as in `UnimodalCTDataset3D`, but automated).
    * Returns survival time and censor indicator (True = censored) exactly
      like the multimodal class.
    """

    def __init__(
        self,
        fold: int,
        split: str,                 # "train" | "test"
        ct_path: str,               # root folder with patient sub-dirs
        labels_splits_path: str,    # TSV with case_id, OS_days, OS_event, fold_X
        vol_std_method: Optional[str] = "padding",
    ):
        super().__init__()
        assert split in ["train", "test"], "`split` must be 'train' or 'test'"

        # --- load split file -------------------------------------------------
        labels_df = pd.read_csv(labels_splits_path, sep="\t")
        split_df = labels_df[labels_df[f"fold_{fold}"] == split][
            ["case_id", "OS_days", "OS_event"]
        ].drop_duplicates("case_id")
        self.labels_df = split_df.reset_index(drop=True)

        self.ct_path = Path(ct_path)
        self.vol_std = VolumeStandardizer(vol_std_method) if vol_std_method else None

        # --- gather CT files -------------------------------------------------
        self.samples: List[Dict] = []
        self.max_depth = 0  # used if padding

        for _, row in self.labels_df.iterrows():
            pid = row.case_id
            patient_dir = self.ct_path / pid
            if not patient_dir.is_dir():
                continue  # patient without CT → skip (rare)

            for ct_file in patient_dir.iterdir():
                if ct_file.suffix != ".npy":
                    continue
                depth = np.load(ct_file, mmap_mode="r").shape[0]
                self.max_depth = max(self.max_depth, depth)

                self.samples.append(
                    {
                        "patient_id": pid,
                        "ct_path": ct_file,
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No CT volumes found for split='{split}' fold={fold} at {ct_path}"
            )

    # --------------------------------------------------------------------- #
    #                            PyTorch hooks                               #
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        pid = sample["patient_id"]

        vol = np.load(sample["ct_path"])
        if self.vol_std is not None:
            vol = self.vol_std(vol, target_depth=self.max_depth)

        vol_tensor = torch.from_numpy(vol).float()  # (D, H, W)

        # ── survival labels ────────────────────────────────────────────────
        row = self.labels_df[self.labels_df.case_id == pid].iloc[0]
        survtime = torch.tensor(row.OS_days, dtype=torch.long)
        censor   = ~torch.tensor(row.OS_event, dtype=torch.bool)  # True := censored

        return {
            "patient_id": pid,
            "volume": vol_tensor,
            "survtime": survtime,
            "censor":   censor,
        }

    # --------------------------------------------------------------------- #
    #                          convenience utils                             #
    # --------------------------------------------------------------------- #
    def stats(self) -> Dict:
        return {
            "total_samples": len(self),
            "total_patients": self.labels_df.case_id.nunique(),
            "max_depth": self.max_depth,
        }

    @staticmethod
    def move_batch_to_device(batch: Dict, device: torch.device) -> None:
        """Move *in-place* every tensor in a batch produced by the DataLoader."""
        batch["volume"]   = batch["volume"].to(device)
        batch["survtime"] = batch["survtime"].to(device)
        batch["censor"]   = batch["censor"].to(device)


# ------------------------------------------------------------------------- #
#                        quick sanity check (optional)                      #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Example usage – adapt the paths / fold numbers to your project layout
    ds = UnimodalCTDatasetSurv(
        fold=0,
        split="test",
        ct_path="./data/processed/processed_CPTAC_PDA_survival/CT",                # <- your root
        labels_splits_path="./data/processed/processed_CPTAC_PDA_survival/k=all.tsv",
        vol_std_method="padding",
    )
    print("Dataset stats:", ds.stats())

    # One sample
    s = ds[0]
    print("patient:", s["patient_id"],
          "| volume shape:", s["volume"].shape,
          "| survtime:", s["survtime"].item(),
          "| censor:", bool(s["censor"]))
