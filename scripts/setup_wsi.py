#External imports
import argparse
import h5py
import os
from PIL import Image
import sys
from pathlib import Path
import logging
from datetime import datetime

#Internal imports
sys.path.insert(0, '../')
from util.data_util import *

def setup_logging():
    """Sets up the logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/setup_wsi_log_{timestamp}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logging.info("Logging setup complete.")


def main(args):
    n_slides_done = 0
    patients = Path(args.patients).read_text().splitlines()
    for svs_file in os.listdir(args.raw_path):
        slide_name = svs_file.split(".")[0]
        patient_id = '-'.join(slide_name.split("-")[:-1]) 
        if patient_id not in patients:
            logging.info(f"{patient_id} not in patients file, skipped.")
            continue
        
        wsi = read_wsi(os.path.join(args.raw_path,slide_name+".svs"))
        if not os.path.exists(os.path.join(args.patch_path,slide_name+".h5")):
            logging.warning(f"No corresponding (.h5) file of {svs_file}, skipped.")
            continue
        
        logging.info(f"saving patches from {svs_file}...")

        with h5py.File(os.path.join(args.patch_path,slide_name+".h5"), 'r') as file:
            dset = file['coords']
            coords = dset[:]
            h5_patch_size = dset.attrs['patch_size']
            h5_patch_level = dset.attrs['patch_level']
        destination_dir_path = os.path.join(args.destination,slide_name)
        os.makedirs(os.path.join(args.destination,slide_name), exist_ok = True)
        for coord in coords:
            patch = wsi.read_region(coord, h5_patch_level, tuple([h5_patch_size, h5_patch_size])).convert('RGB')
            i = 0
            while os.path.exists(os.path.join(destination_dir_path,'%s.png'% i)):
                i += 1
            patch.save(os.path.join(destination_dir_path,'%s.png'% i),"PNG")
        n_slides_done = n_slides_done + 1
        logging.info(f"({n_slides_done}) done {svs_file}.")

    
if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type = str, help = "path of the original raw whole slide images")
    parser.add_argument("--patch_path", type = str, help = "path of the (.h5) coordinates patch folder")
    parser.add_argument("--destination", type = str )
    parser.add_argument("--patients", type = str, help = "path to the (.txt) file containing the list of selected patients")
    args = parser.parse_args()
    
    main(args)
    
    