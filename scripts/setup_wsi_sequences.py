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
            patient_id = patch_name.split("_")[0].split("-")
            patient_id = patient_id[0] + "-" + patient_id[1]
            patch_dict[patient_id].append(patch)

    for patient_id, patches in patch_dict.items():
        seq_num = 0
        if patient_id not in patients:
            logging.info(f"{patient_id} not in patients file, skipped.")
            continue

        for i in range(0, len(patches), 16):
            sequence = patches[i : i + 16]  # noqa : E203
            if len(sequence) < 16:
                break  # Ignore incomplete sequences
            logging.info(f"saving sequence {seq_num} from {patient_id}...")
            seq_folder = os.path.join(args.destination, f"{patient_id}-{seq_num}")
            os.makedirs(seq_folder, exist_ok=True)

            for idx, img in enumerate(sequence):
                src = os.path.join(args.patch_path, img)
                dst = os.path.join(seq_folder, f"{idx}.png")
                shutil.copy(src, dst)

            seq_num += 1
        logging.info(f"{patient_id} done.")


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
