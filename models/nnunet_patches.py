import os 
import numpy as np
from PIL import Image
from scipy.special import softmax
from torch.utils.data import DataLoader

from models.llm import features_extraction
from models.patches import Patches

n = 448
patch_size = 112

dataset_name_conversion = {
    'monuseg': 'Dataset003_MoNuSeg', 
    'panuke': 'Dataset013_PanNuke', 
    'skincancer2': 'Dataset014_skincancer2', 
}


def features_extraction_nnunet(args):
    npz_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
    dataset_dir = os.path.join(args.data_dir, 'nnUNet_raw', dataset_name_conversion[args.dataset])
    images_dir = os.path.join(dataset_dir, 'imagesTe')
    labels_dir = os.path.join(dataset_dir, 'labelsTe')
    predictions_dir = os.path.join(dataset_dir, 'predictionsTe')

    images = sorted([f for f in os.listdir(images_dir) if f.endswith(args.file_ending)])
    labels = sorted([f for f in os.listdir(labels_dir) if f.endswith(args.file_ending)])
    predictions = sorted([f for f in os.listdir(predictions_dir) if f.endswith('.npz')])

    for img, lab, pred in zip(images, labels, predictions):
        # Load image, label, and logits
        image_path = os.path.join(images_dir, img)
        label_path = os.path.join(labels_dir, lab)

        image_array = np.array(Image.open(image_path))[:n, :n]
        label_array = np.array(Image.open(label_path))[:n, :n]
        logits = np.load(os.path.join(predictions_dir, pred), allow_pickle=True)["probabilities"] 

        n_class = logits.shape[0]
        width, height = logits.shape[2], logits.shape[3]

        # Apply softmax to compute probabilities
        probabilities = softmax(logits, axis=0).reshape((n_class, width, height))[:, :n, :n]
        width, height = image_array.shape[:2]

        # Determine cosine similarities between patches in the image
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/pytorch_model.bin")
        tokenizer = get_tokenizer()
        dataset = Patches(image_path, patch_size, preprocess)
        dataloader = DataLoader(dataset, 16)
        args.model = 'conch'
        _, features, _, _, _, _ = features_extraction(args, model, tokenizer, dataset, dataloader, get_position=False)
        cos_sim = features @ features.T

        # Save the numpy arrays in an .npz file
        npz_name = f'{args.dataset}_nnunet_patches_{pred}'
        npz_path = os.path.join(npz_dir, npz_name)
        np.savez_compressed(
            npz_path, 
            width=width, 
            height=height, 
            n_labels=n_class, 
            images=image_array, 
            labels=label_array, 
            probabilities=probabilities, 
            cos_sim=cos_sim
            )

        print(f"Saved features for {img} to {npz_path}")

    return "Feature extraction and saving completed."