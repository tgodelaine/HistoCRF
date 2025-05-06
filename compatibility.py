import numpy as np


def compatibility_from_text_features(args, npz_path):
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    text_features = data['text_features']

    compatibility = text_features @ text_features.T 

    return 1 - compatibility  


def compatibility_from_potts(args, npz_path):
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    n_labels = data['n_labels'] #12
    #weight = args.weight

    compatibility = 1 - np.identity(n_labels)

    return compatibility#weight * compatibility  



