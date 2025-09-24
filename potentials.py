import numpy as np
import torch


device = 'cuda'


def unitary_potential_from_softmax(args, probabilities):
    # Load the .npz file and access the required keys 
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(probabilities) 
    unitary_potential = -(torch.log(probabilities)) 

    return unitary_potential.T     


def pairwise_potential_from_model_features(npz_path, variances, weight, n_components=10): 
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    width = data['width']
    height = data['height'] 
    n = width * height
    features = data['features']

    # Pairwise potential matrix
    pairwise_potential = np.zeros((n,n), dtype=np.float16)
    for i in range(n):
        for j in range(n):
            f_i, f_j = features[i], features[j]
            diff = f_i @ f_j.T 
            if i == j: 
                diff = 0

            pairwise_potential[i, j] = diff 

    return pairwise_potential


def pairwise_potential_from_model_features_and_position(npz_path, variances, weight): 
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    width = data['width']
    height = data['height'] 
    n = width * height
    features = data['features']
    positions = data['positions'].T

    # Pairwise potential matrix
    pairwise_potential = np.zeros((n,n), dtype=np.float16)
    for i in range(n):
        for j in range(n):
            f_i, f_j = features[i], features[j]
            p_i, p_j = positions[i], positions[j]
            diff_f = (f_i - f_j).reshape((-1,1))
            diff_p = (p_i - p_j).reshape((-1,1))

            pairwise_potential[i, j] = (
                + weight[0] * np.exp(-0.5 * (np.linalg.norm(diff_p) / variances[0] + np.linalg.norm(diff_f) / variances[1])) 
                + weight[1] * np.exp(-0.5 * (np.linalg.norm(diff_p) / variances[0]))
            )
    return torch.tensor(pairwise_potential, dtype=torch.float16)