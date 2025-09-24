import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
 

model_clip = timm.create_model(
    "hf-hub:MahmoodLab/UNI",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True,
)

npz_file = '/CECI/home/users/t/g/tgodelai/miccai25/data_processed/skincancer2/nnunet_patches/skincancer2_conch_tile_5824_2688_label_1.npz'
k = 2
patch_idx = 0
patch_size = 112

data = np.load(npz_file)
cos_sim = np.array(data['cos_sim'])

def get_patch_pixel_indices(cos_sim, patch_idx, k=2, patch_size=112, grid_size=4):
    """
    Given patch indices k_patches in a 4x4 grid of 112x112 patches within a 448x448 image,
    return the pixel indices in the original image corresponding to these patches.
    
    Args:
    - k_patches (list of int): Index of patches
    - patch_size (int): Size of each patch (default=112)
    - grid_size (int): Number of patches per row/column (default=4)

    Returns:
    - (set, set): Two sets of pixel index tuples corresponding to patches a and b
    """
    cos_sim_sort = np.argsort(cos_sim, axis=1)
    k_patches = cos_sim_sort[patch_idx, 1:k+1]

    def get_pixel_indices(patch_index):
        row = patch_index // grid_size
        col = patch_index % grid_size

        x_min, y_min = row * patch_size, col * patch_size
        x_max, y_max = x_min + patch_size, y_min + patch_size

        return [(x, y) for x in range(x_min, x_max) for y in range(y_min, y_max)]

    pixel_indices = []
    for patch_idx in k_patches: 
        pixel_indices.append(get_pixel_indices(patch_idx))

    return np.array(sum(pixel_indices, []))

# Example usage
pixel_indices = get_patch_pixel_indices(cos_sim, patch_idx)

def mask_pairwise_potentials(pairwise_potentials, n_pixel, patch_size, cos_sim, num_similar_patches=1):
    """
    Modifies the pairwise potential matrix to keep only intra-patch connections and connections
    to the most similar patches based on cosine similarity.

    Args:
    - pairwise_potentials (np.array): Pairwise potential matrix of shape (n_pixel^2, n_pixel^2).
    - n_pixel (int): Size of the image (assumed square, n_pixel x n_pixel).
    - patch_size (int): Size of each patch.
    - cos_sim (np.array): Cosine similarity matrix of shape (n_patch^2, n_patch^2).
    - num_similar_patches (int): Number of most similar patches to connect.

    Returns:
    - np.array: Modified pairwise potential matrix.
    - np.array: Mask indicating retained connections.
    """

    grid_size = n_pixel // patch_size
    n_patches = grid_size * grid_size  # Total number of patches

    # Step 1: Compute most similar patches using cosine similarity matrix
    similar_patches = {}
    for patch_idx in range(n_patches):
        # Get similarity scores for this patch, ignore self-similarity
        sim_scores = cos_sim[patch_idx].copy()
        sim_scores[patch_idx] = -np.inf  # Prevent self-selection

        # Find the indices of the top `num_similar_patches` patches
        most_similar = np.argsort(sim_scores)[-num_similar_patches:]
        similar_patches[patch_idx] = most_similar.tolist()

    # Step 2: Initialize mask (same shape as pairwise_potentials)
    # mask = np.zeros_like(pairwise_potentials)
    # Step 2: Initialize sparse matrix using LIL format for efficient assignment
    # pairwise_potentials_sparse = sp.lil_matrix((n_pixels, n_pixels), dtype=np.float32)
    # Step 2: Create lists to store sparse tensor indices and values
    indices = []
    values = []

    # Function to get pixel indices of a patch
    def get_patch_pixels(patch_idx):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        x_min, y_min = row * patch_size, col * patch_size
        x_max, y_max = x_min + patch_size, y_min + patch_size
        return [(x, y) for x in range(x_min, x_max) for y in range(y_min, y_max)]

    patch_pixels = {p: get_patch_pixels(p) for p in range(n_patches)}

    def pixel_to_index(x, y):
        return x * n_pixel + y

    # Step 3: Modify the pairwise potential matrix
    for patch_idx, similar_patch_indices in similar_patches.items():
        patch1_indices = [pixel_to_index(x, y) for x, y in patch_pixels[patch_idx]]

        for similar_patch_idx in similar_patch_indices:
            patch2_indices = [pixel_to_index(x, y) for x, y in patch_pixels[similar_patch_idx]]

            # Keep intra-patch connections
            for i in patch1_indices:
                for j in patch1_indices:  
                    #pairwise_potentials_sparse[i, j] = 1  
                    indices.append([i, j])
                    values.append(1.0) 

            # Keep connections to the most similar patch
            for i in patch1_indices:
                for j in patch2_indices:  
                    indices.append([i, j])
                    values.append(1.0) 
                    #pairwise_potentials_sparse[i, j] = 1  

    # Convert lists to tensors
    indices_tensor = torch.tensor(indices, dtype=torch.long).T  # Shape (2, num_nonzero)
    values_tensor = torch.tensor(values, dtype=torch.float32)

    # Create a sparse tensor of shape (n_pixel^2, n_pixel^2)
    sparse_pairwise_potentials = torch.sparse_coo_tensor(indices_tensor, values_tensor, (n_pixels, n_pixels))

    return sparse_pairwise_potentials, similar_patches
    # modified_potentials = pairwise_potentials * mask
    # return modified_potentials, mask, similar_patches
    # return pairwise_potentials_sparse.tocsr(), similar_patches


# Visualization Function
def visualize_matrix(matrix, title="Pairwise Potential Matrix", zoom=None, name_fig='test.png'):
    """
    Visualizes a large matrix using a heatmap.

    Args:
    - matrix (np.array): The matrix to visualize.
    - title (str): Title of the plot.
    - zoom (tuple): If given, zooms into (row_start, row_end, col_start, col_end).
    """

    plt.figure(figsize=(8, 8))
    if zoom:
        row_start, row_end, col_start, col_end = zoom
        matrix = matrix[row_start:row_end, col_start:col_end]

    plt.imshow(matrix, cmap="inferno")#, interpolation="nearest")
    plt.colorbar(label="Connection Strength")
    plt.title(title)
    plt.xlabel("Pixel Index")
    plt.ylabel("Pixel Index")
    plt.show()
    plt.savefig(name_fig)

# Example usage
n_pixel = 4  # Smaller value for visualization
patch_size = 2
n_patches = (n_pixel // patch_size) ** 2  # Total patches

pairwise_potentials = np.random.rand(n_pixel**2, n_pixel**2)  # Simulated pairwise matrix
cos_sim = np.random.rand(n_patches, n_patches)  # Random similarity scores between patches

# Modify the matrix with dynamically computed similar patches
num_similar_patches = 1
modified_potentials, mask, computed_similar_patches = mask_pairwise_potentials(
    pairwise_potentials, n_pixel, patch_size, cos_sim, num_similar_patches
)

# Visualize results
visualize_matrix(mask, title="Masked Pairwise Potential Matrix", name_fig="test1.png")
#zoom_size = 500
#visualize_matrix(mask, title="Zoomed-In View", zoom=(0, zoom_size, 0, zoom_size), name_fig="test2.png")

# Print similar patches for reference
print("Computed Similar Patches:", computed_similar_patches)