import numpy as np
import torch


device = 'cuda'


def compatibility_from_text_features(args, npz_path):
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    text_features = data['text_features']

    compatibility = text_features @ text_features.T 

    return 1 - torch.tensor(compatibility, device=device)


def compatibility_from_potts(args, npz_path):
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    n_labels = data['n_labels'] 

    compatibility = 1 - torch.eye(n_labels, device=device)

    return compatibility 


def compatibility_from_indicator(args, npz_path):
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    n_labels = data['n_labels']

    compatibility = torch.eye(n_labels, device=device)

    return compatibility