import os
from pathlib import Path

# Sanity check of preprocessing


def test_splits():

    data_root = "../data/"
    data_folders = os.listdir(data_root)
    processed_dirs = [
        folder
        for folder in data_folders
        if folder.startswith("processed")
        and os.path.isdir(os.path.join(data_root, folder))
    ]
    for processed in processed_dirs:
        files = os.listdir(os.path.join(data_root, processed))
        # Check that right files are in the directories
        assert "labels.txt" in files
        assert "train.txt" in files
        assert "val.txt" in files
        # assert "CT" in files
        # assert "WSI" in files

        train = (
            Path(os.path.join(data_root, processed, "train.txt"))
            .read_text()
            .splitlines()
        )
        val = (
            Path(os.path.join(data_root, processed, "val.txt")).read_text().splitlines()
        )
        # Check that train and validation don't intersect
        assert len(set(train) & set(val)) == 0
