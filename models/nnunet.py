import os
import numpy as np
from PIL import Image
from scipy.special import softmax

n = 448
patch_size = 112

dataset_name_conversion = {
    'monuseg': 'Dataset003_MoNuSeg', 
    'panuke': 'Dataset013_PanNuke', 
    'skincancer2': 'Dataset254_skincancer2', 
}

def features_extraction(args):
    patch_size = args.patch_size
    print("patch_size", patch_size)
    npz_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
    dataset_dir = os.path.join(args.data_dir, 'nnUNet_raw', dataset_name_conversion[args.dataset])
    images_dir = os.path.join(dataset_dir, 'imagesTe')
    labels_dir = os.path.join(dataset_dir, 'labelsTe')
    predictions_dir = os.path.join(dataset_dir, 'predictionsTe')

    images = sorted([f for f in os.listdir(images_dir) if f.endswith(args.file_ending)])
    labels = sorted([f for f in os.listdir(labels_dir) if f.endswith(args.file_ending)])
    predictions = sorted([f for f in os.listdir(predictions_dir) if f.endswith('.npz')])
    
    for img, lab, pred in zip(images, labels, predictions):
        print(img, lab, pred)
        # Load image, label, and logits
        image_path = os.path.join(images_dir, img)
        label_path = os.path.join(labels_dir, lab)

        image_array = np.array(Image.open(image_path))[:patch_size, :patch_size]
        label_array = np.array(Image.open(label_path))[:patch_size, :patch_size]
        logits = np.load(os.path.join(predictions_dir, pred), allow_pickle=True)["probabilities"] 

        n_class = logits.shape[0]
        width, height = logits.shape[2], logits.shape[3]

        # Apply softmax to compute probabilities
        probabilities = logits.reshape((n_class, width, height))[:, :patch_size, :patch_size]
        #probabilities = softmax(logits, axis=0).reshape((n_class, width, height))[:, :patch_size, :patch_size]
        width, height = image_array.shape[:2]

        # Save the numpy arrays in an .npz file
        npz_name = f'{args.dataset}_{args.patch_size}_{args.model}_{pred}'
        npz_path = os.path.join(npz_dir, npz_name)
        np.savez_compressed(
            npz_path, 
            width=width, 
            height=height, 
            n_labels=n_class, 
            images=image_array, 
            labels=label_array, 
            probabilities=probabilities
            )

        print(f"Saved features for {img} to {npz_path}")

    return "Feature extraction and saving completed."
