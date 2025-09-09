import numpy as np
import time
import torch 

COMPUTE_MEMORY = False


def batched_similarity(features, row_indexes, col_indexes, block_size=1024):
    # Validate index ranges
    values_list = []
    n = row_indexes.shape[0]

    for start in range(0, n, block_size):
        end = min(start + block_size, n)

        rows = features[row_indexes[start:end]]
        cols = features[col_indexes[start:end]]

        block_vals = 1 - torch.sum(rows * cols, dim=1)
        values_list.append(block_vals)

    return torch.cat(values_list, dim=0)


def pairwise_potential_from_model_features(args, npz_path, all_positions, features, variances, data, device='cuda'):
    # Access the required keys
    width = int(data['width']) 
    height = int(data['height']) 

    n_affinity = all_positions.shape[0]
    n_pixels = all_positions.shape[1] 

    features = features.to(device)
    row_indexes = torch.arange(n_affinity, device=device).repeat_interleave(n_pixels)  # Shape (4n,) # Row indexes: (i * torch.ones(len(p), dtype=torch.long))
    col_indexes = all_positions.flatten().to(device)  # Shape (4n,) # Column indexes: Flatten all_positions

    del all_positions

    # Compute squared differences
    values = batched_similarity(features, row_indexes, col_indexes, block_size=5000) #row_indexes.size()[0])

    crow_indexes = torch.zeros(width*height+1, dtype = torch.long, device=device)
    # Compute the count of each index in row_indexes
    counts = torch.bincount(row_indexes, minlength=width * height).to(device)
    # Compute the cumulative sum
    crow_indexes[1:] = torch.cumsum(counts, dim=0)

    if COMPUTE_MEMORY: 
        sparse_memory = (
            crow_indexes.element_size() * crow_indexes.numel() +
            col_indexes.element_size() * col_indexes.numel() +
            values.element_size() * values.numel()
        )/1024**3
        sparse_memory_bits = (
                crow_indexes.element_size() * crow_indexes.numel() +
                col_indexes.element_size() * col_indexes.numel() +
                values.element_size() * values.numel()
            ) * 8
        print(f"[Memory] Sparse potential size (ann): {sparse_memory:.4f} GB, {sparse_memory_bits} bits")

    # Create sparse tensor (coo format)
    sparse_paiwise_potential = torch.sparse_csr_tensor(
        crow_indexes, 
        col_indexes,
        values=values, #/values.max(),
        size=(width*height, width*height),  # Shape of the original affinity matrix
        dtype=torch.float16,
        device=device
    )

    return sparse_paiwise_potential


def pairwise_potential_from_minus_model_features(args, npz_path, all_positions, factors, variances, weight, device='cpu'):
    # Access the required keys
    data = np.load(npz_path)
    width = int(data['width']) 
    height = int(data['height']) 
    features = torch.tensor(data['features'], device=device, dtype=torch.float16)

    n_affinity = all_positions.shape[0]
    n_pixels = all_positions.shape[1] 

    row_indexes = torch.arange(n_affinity).repeat_interleave(n_pixels)  # Shape (4n,) # Row indexes: (i * torch.ones(len(p), dtype=torch.long))
    col_indexes = all_positions.flatten().to(device)  # Shape (4n,) # Column indexes: Flatten all_positions

    # Compute squared differences
    values = (torch.sum(features[row_indexes, :] * features[col_indexes, :], dim=1))

    crow_indexes = torch.zeros(width*height+1, dtype = torch.long, device=device)
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
        print(f"[Memory] Sparse pairwise potential size: {sparse_memory:.4f} GB")
    
    # Create sparse tensor (coo format)
    sparse_paiwise_potential = torch.sparse_csr_tensor(
        crow_indexes, 
        col_indexes,
        values=values, #/values.max(),
        size=(width*height, width*height),  # Shape of the original affinity matrix
        dtype=torch.float16,
        device=device
    )

    return sparse_paiwise_potential


def pairwise_potential_from_minus_model_features_ann(args, npz_path, all_positions, features, variances, weight, device='cpu'):
    # Access the required keys
    data = np.load(npz_path)
    width, height = int(data['width']), int(data['height'])
    
    # convert args.annotations to torch tensor directly
    annotations = torch.tensor(args.annotations, dtype=torch.long, device=all_positions.device)
    annotations_sorted = torch.sort(annotations).values  # replaces np.sort

    n_pixels = all_positions.shape[1]
    n_affinity = all_positions.shape[0]

    # flatten row indices
    row_indexes = all_positions.flatten()  # already a torch tensor

    # repeat sorted annotations for each pixel
    col_indexes = annotations_sorted.repeat_interleave(n_pixels)

    # sort row indices and reindex col indexes
    sorted_row_indexes, sorted_indexes = torch.sort(row_indexes)  # replaces np.argsort
    sorted_col_indexes = col_indexes[sorted_indexes]

    # compute values (dot product similarity)
    values = torch.sum(
        features[row_indexes, :] * features[col_indexes, :], dim=1
    )

    # build crow_indexes with cumulative counts
    crow_indexes = torch.zeros(width * height + 1, dtype=torch.long, device=all_positions.device)
    counts = torch.bincount(sorted_row_indexes, minlength=width * height)
    crow_indexes[1:] = torch.cumsum(counts, dim=0)


    if COMPUTE_MEMORY: 
        sparse_memory = (
            crow_indexes.element_size() * crow_indexes.numel() +
            col_indexes.element_size() * col_indexes.numel() +
            values.element_size() * values.numel()
        )/1024**3
        sparse_memory_bits = (
            crow_indexes.element_size() * crow_indexes.numel() +
            col_indexes.element_size() * col_indexes.numel() +
            values.element_size() * values.numel()
        ) * 8
        print(f"[Memory] Sparse potential size (ann): {sparse_memory:.4f} GB, {sparse_memory_bits} bits")

    # Create sparse tensor (coo format)
    sparse_paiwise_potential = torch.sparse_csr_tensor(
        crow_indexes, 
        sorted_col_indexes,
        values=values, #/values.max(),
        size=(width*height, width*height),  # Shape of the original affinity matrix
        dtype=torch.float16,
        device=device
    )

    args.annotated_row_positions = row_indexes
    
    return sparse_paiwise_potential