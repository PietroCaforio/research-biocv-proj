# External imports
import argparse
import logging
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# Internal imports
sys.path.insert(0, "../")  # noqa: E402


def setup_logging():
    """Sets up the logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/setup_wsi_seq_log_{timestamp}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logging.info("Logging setup complete.")


def main(args):
    patients = Path(args.patients).read_text().splitlines()

    patch_dict = defaultdict(list)

    for patch in sorted(os.listdir(args.patch_path)):
        if patch.endswith(".png"):
            if patch.startswith("._"):
                patch = patch[2:]
            patch_name = patch.split(".")[0]
            patient_id_folder = patch_name.split("_")[0]
            patch_dict[patient_id_folder].append(patch)

    for patient_id_folder, patches in patch_dict.items():
        seq_num = 0
        patient_id = patient_id_folder.split("-")[0]+"-"+patient_id_folder.split("-")[1]
        if patient_id not in patients:
            logging.info(f"{patient_id} not in patients file, skipped.")
            continue
        patch_folder = os.path.join(args.destination, patient_id_folder)
        os.makedirs(patch_folder, exist_ok=True)
        for idx, img in enumerate(patches):
            src = os.path.join(args.patch_path, img)
            dst = os.path.join(patch_folder, f"{idx}.png")
            shutil.copy(src,dst)
        
        logging.info(f"{patch_folder} done.")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_path", type=str, help="path to the patches folder")
    parser.add_argument("--destination", type=str)
    parser.add_argument(
        "--patients",
        type=str,
        help="path to the (.txt) file containing the list of selected patients",
    )
    args = parser.parse_args()

    main(args)
