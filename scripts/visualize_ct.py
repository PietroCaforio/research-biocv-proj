import argparse
import os
import random

import numpy as np
from mayavi import mlab


def choose_random_file(folder_path):
    # List to store all file paths
    all_files = []

    # Walk through the folder and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Append full path of the file
            all_files.append(os.path.join(root, file))

    # Ensure there are files in the directory
    if not all_files:
        return "No files found in the directory!"

    # Choose a random file
    random_file = random.choice(all_files)
    return random_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="../data/processed/CT/")
    args = parser.parse_args()

    # Visualize the 3D volume
    ct_scan = np.load(choose_random_file(args.folder_path))
    mlab.contour3d(ct_scan)

    mlab.show()
