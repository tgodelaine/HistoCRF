import torch
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def get_similar_patches(args, cos_sim):
    """
    For each patch, determine the num_similar_patches similar patches based on the 
    cos_sim matrix

    Args:
    - cos_sim (np.array): Cosine similarity matrix of shape (n_patches^2, n_patches^2).
    - n_patches (int): Number of patches.
    - patch_size (int): Size of each patch.
    - cos_sim (np.array): Cosine similarity matrix of shape (n_patch^2, n_patch^2).
    - num_similar_patches (int): Number of most similar patches to connect.

    Returns:
    - similar_patches (list of list): Each list contains the coordinate of the patch
      and the similar patches associated (e.g. [[((2,2),3,4), ((6,4), 3,3)]]).
    """
    patch_size = args.patch_size
    num_similar_patches = 1 #args.num_similar_patches
    n_patches = int(sqrt(cos_sim.shape[0]))
    similar_patches = []
    grid_size = int(sqrt(n_patches))

    def get_patch_pixels(patch_idx):
        row = int(patch_idx // grid_size * grid_size)
        col = int(patch_idx % grid_size * grid_size)
        return row, col
    
    for patch_idx in range(n_patches):
        sim_scores = cos_sim[patch_idx]
        sim_scores[patch_idx] = -np.inf  # Prevent self-selection

        most_similar = np.argsort(sim_scores)[-num_similar_patches:][0]

        row0, col0 = get_patch_pixels(patch_idx)
        row1, col1 = get_patch_pixels(int(most_similar))
        similar_patches.append([((row0, col0), patch_size, patch_size),((row1, col1), patch_size, patch_size)])

    return similar_patches


def fill_sparse_matrix(image, 
                       variances, weight, 
                       li_patches, 
                       sim_func):
    """
    For each patch, determine the sparse matrix of the pairwise pot based on similar patches

    Args:
    - image (np.array): Cosine similarity matrix of shape (n_patches^2, n_patches^2).
    - variance (list of float):Argument of the pairwise pot.
    - weight (list of float): Argument of the pairwise pot.
    - li_patches (list of list): Each list contains the coordinate of the patch
      and the similar patches associated (e.g. [[((2,2),3,4), ((6,4), 3,3)]]).
    - num_similar_patches (int): Number of most similar patches to connect.

    Returns:
    - csr_mat (torch.tensor): crs sparse tensor of the pairwise pot of shape.
    """
    m,n = image.shape[:2]
    row_indexes = []
    col_indexes = []
    vals = []
    tot_nnz = 0
    
    for jp, (patch1, patch2) in enumerate(li_patches):
        o1, dr1, dc1 = patch1 #origin, delta rows, delta cols # integers
        o2, dr2, dc2 = patch2
        assert dr1 == dr2 
        slice_patch1 = (slice(o1[0], o1[0]+dr1), slice(o1[1], o1[1]+dc1), slice(None))
        slice_patch2 = (slice(o2[0], o2[0]+dr2), slice(o2[1], o2[1]+dc2), slice(None))
        
        patched_image = torch.cat((image[slice_patch1], image[slice_patch2]),
                                  dim = 1)
        patched_sim = sim_func(patched_image, variances, weight)
        #patched_sim /= torch.linalg.norm(patched_sim, dim = -1, keepdims = True) 
        #patched_sim[patched_sim<0.8] = 0 
        coo_patched_sim = patched_sim.to_sparse_coo()
        coo_patched_sim_indexes = coo_patched_sim.indices()
        
        
        patch_row_indexes = []
        patch_row_indexes = []                       
        
        pixel_scalar_indices = []
        for ip in range(dr1):
            ip1 = ip + o1[0]
            for jp in range(o1[1],o1[1]+dc1):
                pixel_scalar_indices.append(ip1*n + jp)
            ip2 = ip + o2[0]
            for jp in range(o2[1],o2[1]+dc2):
                pixel_scalar_indices.append(ip2*n + jp)
        
        patch_row_indexes = coo_patched_sim_indexes[0] #indexes in small similarity matrix
        true_patch_row_indexes = [pixel_scalar_indices[u] for u in patch_row_indexes] #indexes in complete similarity matrix
        row_indexes.append(torch.tensor(true_patch_row_indexes, dtype = torch.int32))
        
        patch_col_indexes = coo_patched_sim_indexes[1]
        true_patch_col_indexes = [pixel_scalar_indices[u] for u in patch_col_indexes] #indexes in complete similarity matrix
        col_indexes.append(torch.tensor(true_patch_col_indexes, dtype = torch.int32))
        
        vals.append(coo_patched_sim.values())
        
        tot_nnz+=patch_col_indexes.shape[0]
        
    
    crow_indexes = torch.zeros(m*n+1, dtype = torch.int32)
    row_indexes = torch.cat(row_indexes)
    col_indexes = torch.cat(col_indexes)
    vals = torch.cat(vals)
    crow_indexes[0] = 0 
    for jrow in range(1,m*n+1):
        crow_indexes[jrow] = crow_indexes[jrow-1] + torch.sum(row_indexes == jrow-1)

    sort_index = torch.argsort(row_indexes)
    col_indexes_sort = col_indexes[sort_index]
    vals_sorted = vals.squeeze()[sort_index]

    csr_mat = torch.sparse_csr_tensor(crow_indexes, 
                                      col_indexes_sort, 
                                      vals_sorted,  
                                      size = ((m*n), (m*n)))
    '''
    coo_mat = torch.sparse_coo_tensor(torch.stack((row_indexes,col_indexes)), 
                                      vals.squeeze(), 
                                      size = ((m*n), (m*n)))
    '''
        
    return csr_mat