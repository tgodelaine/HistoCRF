from datasets import load_dataset
import os
import nibabel as nib
import numpy as np
import json

# Load PanNuke dataset
ds = load_dataset("RationAI/PanNuke")

# Define paths
nnunet_base = "/CECI/proj/medresyst/workshop_trail/data/nnUNet_raw/"
task_name = "Dataset013_PanNuke"
task_path = os.path.join(nnunet_base, task_name)

# Create nnUNet directories
os.makedirs(os.path.join(task_path, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(task_path, "labelsTr"), exist_ok=True)
os.makedirs(os.path.join(task_path, "imagesTs"), exist_ok=True)
os.makedirs(os.path.join(task_path, "labelsTs"), exist_ok=True)

folds = ["fold1", "fold2", "fold3"]

# Save images and labels
for f in folds[:2]:
    for i, example in enumerate(ds[f]):
        # Extract data
        image = example["image"]  # Assuming it's a numpy array (HxWxC)
        mask = example["mask"]    # Assuming it's a numpy array (HxW or HxWxC)

        # Convert to 3D if necessary
        if len(image.shape) == 2:
            image = image[..., np.newaxis]  # Add a channel dimension

        # Save image
        img_path = os.path.join(task_path, "imagesTr", f"image_{i:04d}_0000.nii.gz")  # Single modality (_0000)
        nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)), img_path)

        # Save label
        lbl_path = os.path.join(task_path, "labelsTr", f"label_{i:04d}.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), lbl_path)

for f in folds[2]:
    for i, example in enumerate(ds[f]):
        # Extract data
        image = example["image"]  # Assuming it's a numpy array (HxWxC)
        mask = example["mask"]    # Assuming it's a numpy array (HxW or HxWxC)

        # Convert to 3D if necessary
        if len(image.shape) == 2:
            image = image[..., np.newaxis]  # Add a channel dimension

        # Save image
        img_path = os.path.join(task_path, "imagesTs", f"image_{i:04d}_0000.nii.gz")  # Single modality (_0000)
        nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)), img_path)

        # Save label
        lbl_path = os.path.join(task_path, "labelsTs", f"label_{i:04d}.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), lbl_path)


# Define labels and dataset structure
dataset = {
    "name": "PanNuke",
    "description": "Segmentation dataset for nuclei from 19 tissue types.",
    "tensorImageSize": "4D",
    "modality": {"0": "RGB"},
    "labels": {
        "0": "background",
        "1": "neoplastic_nuclei",
        "2": "inflammatory_nuclei",
        "3": "connective_tissue_nuclei",
        "4": "dead_nuclei",
        "5": "epithelial_nuclei"
    },
    "numTraining": len(ds["train"]),
    "numTest": len(ds["test"]),
    "training": [
        {"image": f"./imagesTr/image_{i:04d}.nii.gz", "label": f"./labelsTr/label_{i:04d}.nii.gz"}
        for i in range(len(ds["train"]))
    ],
    "test": [
        {"image": f"./imagesTs/image_{i:04d}.nii.gz"}
        for i in range(len(ds["test"]))
    ]
}

# Save dataset.json
with open(os.path.join(task_path, "dataset.json"), "w") as f:
    json.dump(dataset, f, indent=4)
