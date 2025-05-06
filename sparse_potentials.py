import numpy as np
import torch
from memory_profiler import profile

COMPUTE_MEMORY = False

#@profile
def pairwise_potential_from_img_and_position(npz_path, all_positions, variances, weight, device='cpu'):

    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width'])
    height = int(data['height'])
    intensity = torch.tensor(data['images'], dtype=torch.float16, device=device)
    intensity /= 255.

    n_affinity = all_positions.shape[0]
    n_pixels = all_positions.shape[1] #n

    # Row indexes: (i * torch.ones(len(p), dtype=torch.long))
    row_indexes = torch.arange(n_affinity).repeat_interleave(n_pixels)  # Shape (4n,)

    # Column indexes: Flatten all_positions
    col_indexes = all_positions.flatten()  # Shape (4n,)

    # Convert i and p to x, y coordinates
    x1, y1 = row_indexes // height, row_indexes % width
    x2, y2 = col_indexes // height, col_indexes % width

    # Compute squared differences
    diff_p = ((x1 - x2)) ** 2 + ((y1 - y2)) ** 2 #NORMALISATION?
    diff_i = torch.sum((intensity[x1, y1]/255 - intensity[x2, y2]/255) ** 2, dim=1)

    # Compute pp values
    pp = (weight[0] * torch.exp(-0.5 * (diff_p / variances[0] + diff_i / variances[1])) +
        weight[1] * torch.exp(-0.5 * (diff_p / variances[0])))

    # Concatenate values
    values = pp

    crow_indexes = torch.zeros(width*height+1, dtype = torch.long)
    # Compute the count of each index in row_indexes
    counts = torch.bincount(row_indexes, minlength=width * height)
    # Compute the cumulative sum
    crow_indexes[1:] = torch.cumsum(counts, dim=0)

    sparse_memory = (
        crow_indexes.element_size() * crow_indexes.numel() +
        col_indexes.element_size() * col_indexes.numel() +
        values.element_size() * values.numel()
    )/1024**3
    print("sparse_memory", sparse_memory)

     # Create sparse tensor (coo format)
    sparse_paiwise_potential = torch.sparse_csr_tensor(
        crow_indexes, 
        col_indexes,
        values=values,
        size=(width*height, width*height),  # Shape of the original affinity matrix
        dtype=torch.float16
    )

    return sparse_paiwise_potential


def pairwise_potential_from_model_features_and_position(args, npz_path, all_positions, factors, variances, weight, device='cpu'):
    mul_factor = 1/factors[0]*torch.ones((factors[0]))
    for f in factors[1:]:
        mul_factor = torch.concat((mul_factor, 1/f*torch.ones((f))))

    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width']) 
    height = int(data['height']) 
    features = torch.tensor(data['features'])

    n_affinity = all_positions.shape[0]
    n_pixels = all_positions.shape[1] 

    # Row indexes: (i * torch.ones(len(p), dtype=torch.long))
    row_indexes = torch.arange(n_affinity).repeat_interleave(n_pixels)  # Shape (4n,)

    # Column indexes: Flatten all_positions
    col_indexes = all_positions.flatten()  # Shape (4n,)

    # Convert i and p to x, y coordinates
    x1, y1 = row_indexes // width, row_indexes % width
    x2, y2 = col_indexes // width, col_indexes % width

    # Compute squared differences
    diff_p = ((x1 - x2)/width) ** 2 + ((y1 - y2)/height) ** 2
    #diff_f = features[row_indexes, :] @ features[col_indexes, :].T # UTILISER ROW_INDEXES OU COL_INDEXES$
    diff_f = torch.sum(features[row_indexes, :] * features[col_indexes, :], dim=1)  #

    # Compute pp values
    pp = (weight[0] * torch.exp(-0.5 * (diff_p / variances[0] + diff_f / variances[1])) +
        weight[1] * torch.exp(-0.5 * (diff_p / variances[0])))

    # Concatenate values
    values_reshaped = pp.view(-1, torch.sum(factors)) 
    values = values_reshaped * mul_factor
    values = torch.tensor(values.flatten())

    crow_indexes = torch.zeros(width*height+1, dtype = torch.long)
    # Compute the count of each index in row_indexes
    counts = torch.bincount(row_indexes, minlength=width * height)
    # Compute the cumulative sum
    crow_indexes[1:] = torch.cumsum(counts, dim=0)

    sparse_memory = (
        crow_indexes.element_size() * crow_indexes.numel() +
        col_indexes.element_size() * col_indexes.numel() +
        values.element_size() * values.numel()
    )/1024**3

     # Create sparse tensor (coo format)
    sparse_paiwise_potential = torch.sparse_csr_tensor(
        crow_indexes, 
        col_indexes,
        values=values, #/values.max(),
        size=(width*height, width*height),  # Shape of the original affinity matrix
        dtype=torch.float16
    )

    return sparse_paiwise_potential


def pairwise_potential_from_prob(args, npz_path, all_positions, factors, variances, weight, device='cpu'):
    first_factors = torch.clone(factors)
    
    if factors[0] > 0 and factors[1] > 0 and factors[2] > 0:
        factors = np.multiply(3, factors)
    elif factors[0] > 0 and factors[1] > 0:
        factors = np.multiply(2, factors)
    elif factors[1] > 0 and factors[2] > 0:
        factors = np.multiply(2, factors)
    elif args.n_affinity[0] > 0 and factors[2] > 0:
        factors = np.multiply(2, factors)

    mul_factor = 1/factors[0]*torch.ones((first_factors[0]))
    for f, ff in zip(factors[1:], first_factors[1:]):
        mul_factor = torch.concat((mul_factor, 1/f*torch.ones((ff))))

    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width']) #MODIFIER WIDTH
    height = int(data['height']) #MODIFIER HEIGHT
    prob = data['probabilities']
    softmax = torch.nn.Softmax(dim=1)
    prob = softmax(torch.tensor(prob / args.temperature))

    n_affinity = all_positions.shape[0]
    n_pixels = all_positions.shape[1] 

    # Row indexes: (i * torch.ones(len(p), dtype=torch.long))
    row_indexes = torch.arange(n_affinity).repeat_interleave(n_pixels)  # Shape (4n,)

    # Column indexes: Flatten all_positions
    col_indexes = all_positions.flatten()  # Shape (4n,)

    # Convert i and p to x, y coordinates
    x1, y1 = row_indexes // height, row_indexes % width
    x2, y2 = col_indexes // height, col_indexes % width

    # Compute squared differences
    diff_p = ((x1 - x2)/width) ** 2 + ((y1 - y2)/height) ** 2
    diff_f = torch.sum(prob[row_indexes, :] * prob[col_indexes, :], dim=1)  #

    #diff_f = torch.sum((features[row_indexes] - features[col_indexes]) ** 2, dim=1)

    # Compute pp values
    pp = (weight[0] * torch.exp(-0.5 * (diff_p / variances[0] + diff_f / variances[1])) +
        weight[1] * torch.exp(-0.5 * (diff_p / variances[0])))

    # Concatenate values
    values_reshaped = pp.view(-1, torch.sum(first_factors)) 
    values = values_reshaped * mul_factor
    values = torch.tensor(values.flatten())

    crow_indexes = torch.zeros(width*height+1, dtype = torch.long)
    # Compute the count of each index in row_indexes
    counts = torch.bincount(row_indexes, minlength=width * height)
    # Compute the cumulative sum
    crow_indexes[1:] = torch.cumsum(counts, dim=0)

    sparse_memory = (
        crow_indexes.element_size() * crow_indexes.numel() +
        col_indexes.element_size() * col_indexes.numel() +
        values.element_size() * values.numel()
    )/1024**3

     # Create sparse tensor (coo format)
    sparse_paiwise_potential = torch.sparse_csr_tensor(
        crow_indexes, 
        col_indexes,
        values=values/values.max(),
        size=(width*height, width*height),  # Shape of the original affinity matrix
        dtype=torch.float16
    )

    return sparse_paiwise_potential


def pairwise_potential_from_model_features(args, npz_path, all_positions, factors, variances, weight, device='cpu'):
    first_factors = torch.clone(factors)
    
    '''
    if factors[0] > 0 and factors[1] > 0 and factors[2] > 0:
        factors = np.multiply(3, factors)
    elif factors[0] > 0 and factors[1] > 0:
        factors = np.multiply(2, factors)
    elif factors[1] > 0 and factors[2] > 0:
        factors = np.multiply(2, factors)
    elif args.n_affinity[0] > 0 and factors[2] > 0:
        factors = np.multiply(2, factors)
    '''

    mul_factor = 1/factors[0]*torch.ones((first_factors[0]))
    for f, ff in zip(factors[1:], first_factors[1:]):
        mul_factor = torch.concat((mul_factor, 1/f*torch.ones((ff))))

    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width']) 
    height = int(data['height']) 
    features = torch.tensor(data['features'])

    n_affinity = all_positions.shape[0]
    n_pixels = all_positions.shape[1] 

    row_indexes = torch.arange(n_affinity).repeat_interleave(n_pixels)  # Shape (4n,) # Row indexes: (i * torch.ones(len(p), dtype=torch.long))
    col_indexes = all_positions.flatten()  # Shape (4n,) # Column indexes: Flatten all_positions

    # Compute squared differences
    diff_f = torch.sum(features[row_indexes, :] * features[col_indexes, :], dim=1)  #
    print(f"diff_f: {torch.mean(diff_f)} +/- {torch.std(diff_f)}")
    # Compute pp values
    pp =  torch.exp(-0.5 * ((1-diff_f) / variances[0]))
    print(f"exp(diff_f): {torch.mean(pp)} +/- {torch.std(pp)}")
    # Concatenate values
    values_reshaped = pp.view(-1, torch.sum(first_factors)) 
    values = values_reshaped * mul_factor
    values = torch.tensor(values.flatten())
    print(f"exp(diff_f)*factor: {torch.mean(values)} +/- {torch.std(values)}")

    '''
    # Compute squared differences
    diff_f = torch.sum(features[row_indexes, :] * features[col_indexes, :], dim=1) 
    diff_f_reshaped = diff_f.view(-1, torch.sum(first_factors)) 
    diff_f = diff_f_reshaped * mul_factor
    diff_f = torch.tensor(diff_f.flatten())
    values = diff_f
    print(f"diff_f*factor: {torch.mean(values)} +/- {torch.std(values)}")
    '''

    crow_indexes = torch.zeros(width*height+1, dtype = torch.long)
    # Compute the count of each index in row_indexes
    counts = torch.bincount(row_indexes, minlength=width * height)
    # Compute the cumulative sum
    crow_indexes[1:] = torch.cumsum(counts, dim=0)

    if COMPUTE_MEMORY: 
        sparse_memory = (
            crow_indexes.element_size() * crow_indexes.numel() +
            col_indexes.element_size() * col_indexes.numel() +
            values.element_size() * values.numel()
        )/1024**3
        print("sparse_memory", sparse_memory)

    # Create sparse tensor (coo format)
    sparse_paiwise_potential = torch.sparse_csr_tensor(
        crow_indexes, 
        col_indexes,
        values=values, #/values.max(),
        size=(width*height, width*height),  # Shape of the original affinity matrix
        dtype=torch.float16
    )

    #args.weight[1] = torch.mean(values)

    return sparse_paiwise_potential