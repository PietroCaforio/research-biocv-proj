import os
from pathlib import Path
import sys
import pytest


sys.path.insert(0, '../')
from data.unimodal import *
from data.unimodal3D import *


# Sanity check of preprocessing
def test_splits():
    data_root = "../data/"
    data_folders = os.listdir(data_root)
    processed_dirs = [folder for folder in data_folders if folder.startswith("processed") and os.path.isdir(os.path.join(data_root, folder))]
    for processed in processed_dirs:
        files = os.listdir(os.path.join(data_root, processed))
        # Check that right files are in the directories
        assert "labels.txt" in files
        assert "train.txt" in files
        assert "val.txt" in files
        #assert "CT" in files
        #assert "WSI" in files
        
        train = Path(os.path.join(data_root, processed, "train.txt")).read_text().splitlines()        
        val = Path(os.path.join(data_root, processed, "val.txt")).read_text().splitlines()
        # Check that the number of entries in the files is equal to the number of patients actually present in the dataset
        #assert len(os.listdir(os.path.join(data_root, processed, "CT"))) == len(train) + len(val)
        # Check that train and validation don't intersect
        assert len(set(train) & set(val)) == 0
    
# Parametrized test for UnimodalCTDataset
@pytest.mark.parametrize("unimodal", [
    UnimodalCTDataset(split="all", dataset_path="../data/processed"),
    UnimodalCTDataset(split="val", dataset_path="../data/processed"),
    UnimodalCTDataset(split="train", dataset_path="../data/processed"),
    UnimodalCTDataset(split="val", dataset_path="../data/processed_oversampling"),
    UnimodalCTDataset(split="train", dataset_path="../data/processed_oversampling"),
])
def test_unimodal_dataset(unimodal):  # Updated the function name to avoid confusion
    dataset = unimodal
    # Check stats of dataset
    dataset.stats()

    # Check the first few items in the dataset
    for i in range(3):
        min = 0
        max = len(dataset)
        item = dataset[randint(min, max-1)]  # Fixed index access: max-1

    # DataLoader check
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Get a batch of data
    batch = next(iter(dataloader))

    # Check the batch contents
    assert "patient_id" in batch
    assert "frame" in batch
    assert "label" in batch

    # Move batch to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.move_batch_to_device(batch, device)
    

# Parametrized test for UnimodalCTDataset3D
@pytest.mark.parametrize("unimodal3D", [
    UnimodalCTDataset3D(split="all", dataset_path="../data/processed"),
    UnimodalCTDataset3D(split="val", dataset_path="../data/processed"),
    UnimodalCTDataset3D(split="train", dataset_path="../data/processed"),
    UnimodalCTDataset3D(split="val", dataset_path="../data/processed_oversampling"),
    UnimodalCTDataset3D(split="train", dataset_path="../data/processed_oversampling"),
])
def test_unimodal3D_dataset(unimodal3D):  # Updated the function name to avoid confusion
    dataset = unimodal3D
    # Check stats of dataset
    dataset.stats()

    # Check the first few items in the dataset
    for i in range(3):
        min = 0
        max = len(dataset)
        item = dataset[randint(min, max-1)]  # Fixed index access: max-1

    # DataLoader check
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Get a batch of data
    batch = next(iter(dataloader))

    # Check the batch contents
    assert "patient_id" in batch
    assert "label" in batch

    # Move batch to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.move_batch_to_device(batch, device)

