import torch

device = 'cuda'

def compatibility_from_text_features(args, data):
    # Access the required keys
    text_features = torch.tensor(data['text_features'], device=device)

    compatibility = 1 - text_features @ text_features.T 

    return compatibility


def compatibility_from_potts(args, data):
    # Access the required keys
    n_labels = data['n_labels']

    compatibility = 1 - torch.eye(n_labels, device=device)

    return compatibility 


def compatibility_from_indicator(args, data):
    # Access the required keys
    n_labels = data['n_labels']

    compatibility = torch.eye(n_labels, device=device)

    return compatibility