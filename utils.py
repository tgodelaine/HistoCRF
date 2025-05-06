from math import sqrt
import numpy as np 
from PIL import Image
from scipy.ndimage import label
import torch
from memory_profiler import profile

def dice(pred, target):
    pred = pred.ravel()  # Flatten arrays if necessary
    target = target.ravel()
    intersection = (pred * target).sum()  # Assuming binary labels (0 and 1)
    return 2.0 * intersection / (pred.sum() + target.sum() + 1e-7)  # Adding epsilon to avoid division by zero


def obtain_cell_of_interest_mask(file_path, cell_of_interest=1):
    # Obtain segmentation mask 
    file = Image.open(file_path)
    binary_mask = np.array(file)

    # Label the connected components
    labeled_mask, num_features = label(binary_mask) # binary_mask is your binary segmentation mask

    # Select the cell of interest 
    cell_mask = (labeled_mask == cell_of_interest)  # Replace 1 with the desired label

    return cell_mask


def obtain_patch_of_interest_mask(file_path, patch_of_interest=[1], patch_size=25, oned=False):
    # Obtain segmentation mask 
    file = np.load(file_path)
    binary_mask = file["labels"]
    
    if not oned: 
        binary_mask_width, binary_mask_height = binary_mask.shape

        # Obtain labeled mask
        labeled_mask = np.ones((binary_mask_width, binary_mask_height))*(-1)
        n_patch_h = binary_mask_height // patch_size
        n_patch_w = binary_mask_width // patch_size
        for poi in patch_of_interest: 
            row = poi // n_patch_h
            column = poi - (row * n_patch_w) 
            labeled_mask[column*patch_size:(column+1)*patch_size, row*patch_size:(row+1)*patch_size] = binary_mask[column*patch_size:(column+1)*patch_size, row*patch_size:(row+1)*patch_size]
    
    else:
        binary_mask_width, binary_mask_height = binary_mask.shape[0], 1

        # Obtain labeled mask
        labeled_mask = np.ones((binary_mask_width, binary_mask_height))*(-1)
        n_patch_w = binary_mask_width // patch_size
        for poi in patch_of_interest: 
            row = 0
            column = poi
            labeled_mask[column:(column+1)] = binary_mask[column:(column+1)]
    
    return labeled_mask 


def save_npz(labeled_masks):
    npz_path = 'labeled_masks.npz'
    np.savez(npz_path, labeled_masks)
    return npz_path


#ajouter dans args, patch_idx, k, patch_size, grid_size
def get_patch_pixel_indices(args, cos_sim, patch_idx):
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
    k_patches = cos_sim_sort[patch_idx, 1:args.k+1]

    def get_pixel_indices(patch_index):
        row = patch_index // args.grid_size
        col = patch_index % args.grid_size

        x_min, y_min = row * args.patch_size, col * args.patch_size
        x_max, y_max = x_min + args.patch_size, y_min + args.patch_size

        return [(x, y) for x in range(x_min, x_max) for y in range(y_min, y_max)]

    pixel_indices = []
    for patches_idx_ in k_patches: 
        pixel_indices.append(get_pixel_indices(patches_idx_))

    return np.array(sum(pixel_indices, []))


def optimize_pixel_selection(img, image_quantization, num_selected=2, random=False):
    def find_neighbors(x, y, img_size, n_neighbors):
        # Local (4-connectivity neighbors)
        if n_neighbors == 4: 
            neighbors = np.array([[x-1, y], [x, y+1], [x+1, y], [x, y-1]])
        elif n_neighbors == 8: 
            neighbors = np.array([[x-1, y], [x, y+1], [x+1, y], [x, y-1], [x-1,y-1], [x-1, y+1], [x+1, y-1], [x+1, y+1]])
        elif n_neighbors == 50:
            def get_neighbors(radius=3):
                neighbors = []
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:  # Exclude the center pixel itself
                            continue
                        neighbors.append([dx, dy])
                return np.array(neighbors)
            neighbors =  get_neighbors(5)
        else: 
            raise RuntimeError(f"Wrong value {n_neighbors} for argument n_neighbors")
        neighbors = np.clip(neighbors, 0, img_size-1)  

        return neighbors
    img_size = img.shape[0]

    # Precompute pixel groupings
    unique_values = np.unique(image_quantization)
    group_positions = {val: np.column_stack(np.where(image_quantization == val)) for val in unique_values}

    def find_positions_of_selected_pixel(x, y, img_size, image_quantization, group_positions, num_selected=2):
        pixel = image_quantization[x, y]

        # Non-local (Select random pixels from the same group)
        if not random:
            pixel_positions = group_positions[pixel]  # Precomputed pixel locations
            random_indices = np.random.choice(len(pixel_positions), num_selected, replace=False)
            selected_pixel_positions = pixel_positions[random_indices]
        else:
            row_indexes = np.random.randint(0, img_size, (num_selected, 1))
            col_indexes = np.random.randint(0, img_size, (num_selected, 1))
            selected_pixel_positions = np.concatenate((row_indexes, col_indexes), axis=1)

        # Local (4-connectivity neighbors)
        #neighbors = np.array([[x-1, y], [x, y+1], [x+1, y], [x, y-1]])
        #neighbors = np.clip(neighbors, 0, img_size-1)  # Ensure valid indices
        neighbors = find_neighbors(x, y, img_size, num_selected)

        # Combine non-local and local selections
        all_positions = np.vstack((selected_pixel_positions, neighbors))
        return all_positions[:, 0] * img_size + all_positions[:, 1]

    # Vectorized position computation
    all_positions = [
        find_positions_of_selected_pixel(x, y, img_size, image_quantization, group_positions, num_selected)
        for x in range(img_size) for y in range(img_size)
    ]
    
    return np.array(all_positions)


def optimize_pixel_selection_neighbors(img_size, n_neighbors):
    def find_neighbors(x, y, img_size, n_neighbors):
        # Local (4-connectivity neighbors)
        if n_neighbors == 4: 
            neighbors = np.array([[x-1, y], [x, y+1], [x+1, y], [x, y-1], [x-1, y-1]])
        elif n_neighbors == 8: 
            neighbors = np.array([[x-1, y], [x, y+1], [x+1, y], [x, y-1], [x-1,y-1], [x-1, y+1], [x+1, y-1], [x+1, y+1]])
        elif n_neighbors == 50:
            def get_neighbors(radius=3):
                neighbors = []
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:  # Exclude the center pixel itself
                            continue
                        neighbors.append([dx, dy])
                return np.array(neighbors)
            neighbors =  get_neighbors(5)
        else: 
            raise RuntimeError(f"Wrong value {n_neighbors} for argument n_neighbors")
        neighbors = np.clip(neighbors, 0, img_size-1)  

        # Combine non-local and local selections
        return neighbors[:, 0] * img_size + neighbors[:, 1]

    # Vectorized position computation
    all_positions = [
        find_neighbors(x, y, img_size, n_neighbors)
        for x in range(img_size) for y in range(img_size)
    ]
    
    return np.array(all_positions)


def convert_to_sparse(args, pairwise_potential, img=None):
    m, n = pairwise_potential.size()

    zero_method = args.zero_method
    threshold = args.threshold
    n_affinity = args.n_affinity
    if zero_method == 'random':
        row_indexes, col_indexes = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        values = torch.empty(0)
        for i in range(m):
            row_indexes_i = i*torch.ones(n_affinity, dtype=torch.long)
            col_indexes_i = torch.randint(0, n, (n_affinity,), dtype=torch.long)
            row_indexes = torch.cat([row_indexes, row_indexes_i])
            col_indexes = torch.cat([col_indexes, col_indexes_i])
            values = torch.cat([values, pairwise_potential[row_indexes_i, col_indexes_i]])

    elif zero_method == 'threshold': 
        row_indexes, col_indexes = torch.where (pairwise_potential > threshold)
        values = pairwise_potential[pairwise_potential > threshold]

    elif zero_method == 'neighbor':
        top_values, top_indices = torch.topk(pairwise_potential, k=n_affinity, dim=1) # Get the top k values and their indices per row
        row_indexes = torch.arange(m).repeat_interleave(n_affinity) # Create row indices (i.e., which pixel the affinities belong to)
        col_indexes = top_indices.flatten()
        values = top_values.flatten()

    elif zero_method == 'nonlocal':
        image_quantization = np.array(Image.fromarray(img).quantize(64))
        all_positions = optimize_pixel_selection(img, image_quantization, args.n_affinity)
        rows = np.arange(img.shape[0]**2)[:, None]
        sparse_paiwise_potential = torch.zeros_like(pairwise_potential)
        sparse_paiwise_potential[rows, all_positions] = pairwise_potential[rows, all_positions]
        #pairwise_potential[rows, all_positions] = 0
    
    elif zero_method == 'localneighbor':
        img_size = int(sqrt(m))
        all_positions = optimize_pixel_selection_neighbors(img_size, args.n_affinity)
        rows = np.arange(m)[:, None]
        sparse_paiwise_potential = torch.zeros_like(pairwise_potential)
        sparse_paiwise_potential[rows, all_positions] = pairwise_potential[rows, all_positions]

    elif zero_method == 'nonlocalrandom':
        image_quantization = np.array(Image.fromarray(img).quantize(64))
        all_positions = optimize_pixel_selection(img, image_quantization, args.n_affinity, True)
        rows = np.arange(img.shape[0]**2)[:, None]
        sparse_paiwise_potential = torch.zeros_like(pairwise_potential)
        sparse_paiwise_potential[rows, all_positions] = pairwise_potential[rows, all_positions]

    elif zero_method == 'randomsparse':
        row_indexes, col_indexes = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        values = torch.empty(0)
        for i in range(m):
            row_indexes_i = i*torch.ones(n_affinity, dtype=torch.long)
            col_indexes_i = torch.randint(0, n, (n_affinity,), dtype=torch.long)
            row_indexes = torch.cat([row_indexes, row_indexes_i])
            col_indexes = torch.cat([col_indexes, col_indexes_i])
            values = torch.cat([values, pairwise_potential[row_indexes_i, col_indexes_i]])
    
    else: 
        raise RuntimeError(f"Incorrect value {zero_method} for zero_method.")
    
    if zero_method in ['random', 'threshold', 'neighbor']: 
        crow_indexes = torch.zeros(m+1, dtype = torch.long)
        crow_indexes[0] = 0 
        for jrow in range(1,m+1):
            crow_indexes[jrow] = crow_indexes[jrow-1] + torch.sum(row_indexes == jrow-1)

        # Create sparse tensor (coo format)
        sparse_paiwise_potential = torch.sparse_csr_tensor(
            crow_indexes, 
            col_indexes,
            values=values,
            size=(m, n),  # Shape of the original affinity matrix
            dtype=torch.float16
        )
        return sparse_paiwise_potential.to_dense()
    elif zero_method in ['randomsparse']: 
        crow_indexes = torch.zeros(m+1, dtype = torch.long)
        crow_indexes[0] = 0 
        for jrow in range(1,m+1):
            crow_indexes[jrow] = crow_indexes[jrow-1] + torch.sum(row_indexes == jrow-1)

        # Create sparse tensor (coo format)
        sparse_paiwise_potential = torch.sparse_csr_tensor(
            crow_indexes, 
            col_indexes,
            values=values/values.max(),
            size=(m, n),  # Shape of the original affinity matrix
            dtype=torch.float16
        )
        return sparse_paiwise_potential 
    else:
        return sparse_paiwise_potential
 

def optimize_pixel_selection_neighbors_vect(img_size, n_neighbors):
    # Generate all pixel coordinates
    x, y = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]), indexing="ij")

    # Define neighbor offsets based on n_neighbors
    if n_neighbors == 4:
        offsets = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    elif n_neighbors == 8:
        offsets = np.array([
            [-1, 0], [0, 1], [1, 0], [0, -1], [-1, -1], [-1, 1], [1, -1], [1, 1]
        ])
    elif n_neighbors == 16:
        radius = 2
        dx, dy = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij")
        offsets = np.column_stack([dx.ravel(), dy.ravel()])
        offsets = offsets[~(offsets == 0).all(axis=1)]  # Remove center (0,0)
        rng = np.random.default_rng(42) # Fix randomness
        indices = np.linspace(0, len(offsets) - 1, 16, dtype=int) # Evenly spaced selection
        offsets = offsets[indices]
    elif n_neighbors == 50:
        radius = 4
        dx, dy = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij")
        offsets = np.column_stack([dx.ravel(), dy.ravel()])
        offsets = offsets[~(offsets == 0).all(axis=1)]  # Remove center (0,0)
        rng = np.random.default_rng(42) # Fix randomness
        indices = np.linspace(0, len(offsets) - 1, 50, dtype=int) # Evenly spaced selection
        offsets = offsets[indices]
    else:
        raise RuntimeError(f"Wrong value {n_neighbors} for argument n_neighbors")

    # Compute all neighbors in a vectorized way
    neighbors_x = x[..., None] + offsets[:, 0]  # Broadcast x positions
    neighbors_y = y[..., None] + offsets[:, 1]  # Broadcast y positions

    # Clip to ensure in-bounds indices
    neighbors_x = np.clip(neighbors_x, 0, img_size[0] - 1)
    neighbors_y = np.clip(neighbors_y, 0, img_size[1] - 1)

    # Convert to linear indices
    all_positions = neighbors_x * img_size[1] + neighbors_y  # Shape: (img_size, img_size, num_neighbors)

    return torch.tensor(all_positions.reshape(img_size[0]*img_size[1], -1),  dtype=torch.long) 


def optimize_pixel_selection_vect(args, img, npz_path, prob=False):
    num_selected = args.n_affinity[1]
    data = np.load(npz_path)
    img_size = data["width"], data["height"]

    labels = data['labels']
    if prob:
        probabilities = data['probabilities']
        labels = np.argmax(probabilities, axis=1)

    # Non local connections
    selected_non_local_pixels = np.empty([0,num_selected+1])
    unique_values = np.unique(labels)  # Get unique values in the quantized image and group pixel coordinates accordingly
    for val in unique_values:
        group_positions = np.where(labels == val)[0]
        
        num_pixels_in_group = group_positions.shape[0]

        random_indices = np.random.randint(0, num_pixels_in_group, (num_pixels_in_group * num_selected))
        random_positions = np.reshape(group_positions[random_indices], (num_pixels_in_group, num_selected))

        random_positions = np.concatenate((np.expand_dims(group_positions, axis=-1), random_positions), axis=1)

        selected_non_local_pixels = np.concatenate((selected_non_local_pixels, random_positions), axis=0)

    sorting = np.argsort(selected_non_local_pixels[:, 0])
    non_local_indices = selected_non_local_pixels[:,1:][sorting]

    # Combine local and non-local selections
    all_positions = non_local_indices

    return torch.tensor(all_positions.reshape(img_size[0]*img_size[1], -1),  dtype=torch.long)


def cossim_pixel_selection(args, img, npz_path):
    num_selected = args.n_affinity[1]
    data = np.load(npz_path)
    img_size = data["width"], data["height"]

    cossim = data["features"]@data["features"].T
    #indices_ordered = np.argsort(cossim, axis=1)[::-1]
    #indices_selected = indices_ordered[:, :args.n_affinity[1]]
    print("cossim", cossim[cossim > 0.7])
    indices_high_affinities = cossim > 0.7
    rng = np.random.default_rng(seed=42)
    selected_indices = np.array([
        rng.choice(np.flatnonzero(row), size=args.n_affinity[1], replace=True)
        for row in indices_high_affinities
    ])
    return torch.tensor(selected_indices,  dtype=torch.long)


def annotation_pixel_selection(args, npz_path):
    num_selected = args.n_annotations
    annotations_class = get_annotation(args, npz_path, return_class=True)
    annotations = np.array(args.annotations)
    #annotations = np.array(annotations)
    data = np.load(npz_path)
    img_size = data["width"], data["height"]
    labels = data['labels']
    #probabilities = data['probabilities']
    #labels = np.argmax(probabilities, axis=1)

    # Non local connections
    selected_non_local_pixels = np.empty([0,num_selected+1])
    unique_values = np.unique(labels)  # Get unique values in the quantized image and group pixel coordinates accordingly
    for val in unique_values:
        group_positions = np.where(labels == val)[0]
        num_pixels_in_group = group_positions.shape[0]

        annotation_positions = np.where(annotations_class == val)[0]
        if len(annotation_positions) == 0:
            annotation_indices = [[item] * num_selected for item in group_positions]
        else:
            annotation_indices = [annotations[annotation_positions]] * num_pixels_in_group
        #random_positions = np.reshape(group_positions[random_indices], (num_pixels_in_group, num_selected))
        random_positions = np.concatenate((np.expand_dims(group_positions, axis=-1), annotation_indices), axis=1)
        selected_non_local_pixels = np.concatenate((selected_non_local_pixels, random_positions), axis=0)

    sorting = np.argsort(selected_non_local_pixels[:, 0])
    non_local_indices = selected_non_local_pixels[:,1:][sorting]

    # Combine local and non-local selections
    all_positions = non_local_indices

    return torch.tensor(all_positions.reshape(img_size[0]*img_size[1], -1),  dtype=torch.long)


#@profile
def determine_sparsity(args, n_pixel, npz_path, img=None):
    annotated_positions = args.annotations 
    factors = torch.tensor([0, 0, 0])

    sparse_method = args.sparse_method

    img_size = np.load(npz_path)["width"], np.load(npz_path)["height"]

    all_positions = None

    if args.n_affinity[0] > 0:
        all_positions = optimize_pixel_selection_neighbors_vect(img_size, args.n_affinity[0])
        factors[0] = args.n_affinity[0]

    if args.n_affinity[1] > 0: 
        factors[1] = args.n_affinity[1]
        if sparse_method == 'random':
            all_positions_non_local = torch.tensor(torch.randint(low=0, high=n_pixel, size=(n_pixel, args.n_affinity[1])), dtype=torch.long)
        elif sparse_method == 'oracle':
            all_positions_non_local = optimize_pixel_selection_vect(args, img, npz_path)
        elif sparse_method == 'zsprob':
            all_positions_non_local = optimize_pixel_selection_vect(args, img, npz_path, prob=True)
        elif sparse_method == 'cossim':
            all_positions_non_local = cossim_pixel_selection(args, img, npz_path)
        
        if args.n_affinity[0] > 0:
            all_positions = torch.concat((all_positions, all_positions_non_local), axis=1)
        else:
            all_positions = all_positions_non_local
    
    elif args.n_affinity[1] > 0 and args.n_affinity[0] == 0: 
        all_positions = torch.tensor(torch.randint(low=0, high=n_pixel, size=(n_pixel, args.n_affinity[1])), dtype=torch.long)

    if (args.n_affinity[0] > 0 or args.n_affinity[1] > 0): 
        if len(annotated_positions) > 0:
            #all_positions_annotations = annotation_pixel_selection(args, npz_path)
            #factors[2] = args.n_annotations
            all_positions_annotations = torch.tensor(annotated_positions).repeat((all_positions.size()[0],1))
            factors[2] = len(args.annotations) 
            all_positions = torch.concat((all_positions, all_positions_annotations), axis=1)

    elif len(annotated_positions) > 0:
        #all_positions = annotation_pixel_selection(args, npz_path)
        #factors[2] = args.n_annotations
        all_positions = torch.tensor(annotated_positions).repeat((img_size[0]*img_size[1],1))
        factors[2] = len(args.annotations) 

    return all_positions, factors


def annotated_unitary_potential(args, npz_path, unitary_potential):
    annotated_positions = args.annotations 
    labels = np.load(npz_path)['labels']
    annotated_labels = labels[annotated_positions]
    for annotated_position, annotated_label in zip(annotated_positions, annotated_labels):
        new_unitary_pot = torch.zeros((unitary_potential.size()[0]))
        new_unitary_pot[annotated_label] = 1
        softmax = torch.nn.Softmax(dim=0)
        new_unitary_pot = - torch.log(softmax(new_unitary_pot / args.temperature))
        unitary_potential[:, annotated_position] = new_unitary_pot
    return unitary_potential


def get_annotation(args, npz_path, return_class=False):
    data = np.load(npz_path)
    labels = data['labels']
    n_class = np.unique(labels)
    annotations = []
    label_annotations = []
    if return_class:
        for c in n_class:
            label_annotations.extend([c]*args.n_annotations)
        return label_annotations
    for c in n_class:
        rng = np.random.default_rng(seed=42)
        c_labels = np.where(labels == c)[0]  #np.where(labels == c)[0] #random.shuffle(np.where(labels == c)[0])
        c_labels = rng.permutation(c_labels)
        if len(c_labels) < args.n_annotations:
            annotations.extend(c_labels)
            annotations.extend([c_labels[0]]*(args.n_annotations-len(c_labels)))
        else:
            annotations.extend(c_labels[:args.n_annotations])
        label_annotations.extend([c]*args.n_annotations)
    return annotations