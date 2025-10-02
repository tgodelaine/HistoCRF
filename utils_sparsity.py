import numpy as np 
import time
import torch

from utils_annotations import get_annotation

device = 'cuda'


def determine_sparsity_split_split(args, n_pixel, npz_path, img=None, features=None, cossim=None):
    annotated_positions = args.annotations 
    factors = torch.tensor([0, 0, 0])

    sparse_method = args.sparse_method

    data = np.load(npz_path)
    img_size = data["width"], data["height"]

    local_positions, nonlocal_positions, annotation_positions = None, None, None

    if args.n_affinity[0] > 0:
        local_positions = optimize_pixel_selection_neighbors_vect(img_size, args.n_affinity[0])
        factors[0] = args.n_affinity[0]

    if args.n_affinity[1] > 0: 
        factors[1] = args.n_affinity[1]
        if sparse_method == 'random':
            torch.manual_seed(args.seed + args.it + args.crf_it)
            nonlocal_positions = torch.tensor(torch.randint(low=0, high=n_pixel, size=(n_pixel, args.n_affinity[1])), dtype=torch.long)
        elif sparse_method == 'oracle':
            nonlocal_positions = optimize_pixel_selection_vect(args, img, npz_path)
        elif sparse_method == 'zsprob':
            nonlocal_positions = optimize_pixel_selection_vect(args, img, npz_path, prob=True)
        elif sparse_method == 'cossim':
            nonlocal_positions = cossim_pixel_selection(args, npz_path, cossim)

    if len(annotated_positions) > 0:
        annotation_positions = annotation_pixel_selection_cossim(args, npz_path, features) #Seule les images ayant le même labels prédits sont reliées aux images annotées de 
        #annotation_positions = torch.tensor(annotated_positions).repeat((img_size[0]*img_size[1],1)) #Toutes les images sont reliées aux images annotées
        factors[2] = len(args.annotations) 

    return local_positions, nonlocal_positions, annotation_positions, factors


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


def cossim_pixel_selection(args, npz_path, padded, max=False):
    torch.manual_seed(args.it)
    np.random.seed(args.it)

    rand_idx = torch.randint(0, padded.size()[1], (padded.size()[0], args.n_affinity[1]), device=padded.device)
    indices_selected = torch.gather(padded, 1, rand_idx).to(torch.long)

    return indices_selected 


def annotation_pixel_selection_cossim(args, npz_path, features): #Liaison avec les patchs qui ont les features les plus similaires
    #Chaque patch annoté est liés aux patchs les plus proches 
    random = True

    annotations = torch.tensor(args.annotations, dtype=torch.long, device=device)
    annotations = torch.sort(annotations).values  # sorted annotations

    # features is already a torch.Tensor
    diff_f = features[annotations, :] @ features.T   # (n_annotations, n_samples)
    diff_f_sorted = torch.argsort(diff_f, dim=1)    # torch version of np.argsort

    n_affinity = 5
    n_affinity_random = 2 * n_affinity
    
    if random:
        diff_f_selected_random = diff_f_sorted[:, -n_affinity_random:]  # shape (n_annotations, 2*n_affinity)

        # use torch.randint or multinomial instead of numpy RNG
        idx = torch.randint(
            low=0,
            high=diff_f_selected_random.size(1),
            size=(diff_f_selected_random.size(0), n_affinity),
            device=features.device
        )
        diff_f_selected = torch.gather(diff_f_selected_random, 1, idx)

    else:
        diff_f_selected = diff_f_sorted[:, -n_affinity:]

    return diff_f_selected.long()


def annotation_pixel_selection_color(args, npz_path, unitary_potential): 
    #Liaison avec les pixels de même quantile de couleur

    num_selected = args.n_annotations
    annotations_class = get_annotation(args, npz_path, return_class=True)
    annotations = np.array(args.annotations)

    data = np.load(npz_path)
    img_size = data["width"], data["height"]
    labels = data['labels']

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


def annotation_pixel_selection_prediction(args, npz_path, unitary_potential): 
    #Liaison avec les patchs dont le label prédit est le même que l'annotation
    data = np.load(npz_path)
    img_size = data["width"], data["height"]
    labels = data['labels']
    #probabilities = data['probabilities']

    annotations = np.array(args.annotations)
    annotations_labels = labels[annotations]
    n_annotations = len(annotations)

    #predicted_labels = np.argmax(probabilities, axis=1)
    predicted_labels = np.argmin(unitary_potential, axis=0)

    indices_selected = np.empty([0, n_annotations+1])

    unique_labels = np.unique(predicted_labels) 
    for l in unique_labels:
        indices_predictions_l = np.where(predicted_labels == l)[0]
        n_pixels_in_group = indices_predictions_l.shape[0]

        indicies_annotations_l = np.where(annotations_labels == l)[0]
        if len(indicies_annotations_l) == 0:
            annotation_indices = [[-1] * n_annotations for idx in indices_predictions_l] #Si aucune image annotée n'appartient à la classe l, on lie juste les images avec elles mêmes 
        else:
            annotation_indices = np.concatenate((np.array([annotations[indicies_annotations_l]] * n_pixels_in_group), np.array([[-1] * (n_annotations-len(indicies_annotations_l)) for idx in indices_predictions_l])), axis=1)
        random_positions = np.concatenate((np.expand_dims(indices_predictions_l, axis=-1), annotation_indices), axis=1)
        indices_selected = np.concatenate((indices_selected, random_positions), axis=0)

    sorting = np.argsort(indices_selected[:, 0])
    non_local_indices = indices_selected[:,1:][sorting]

    # Combine local and non-local selections
    all_positions = non_local_indices

    return torch.tensor(all_positions.reshape(img_size[0]*img_size[1], -1),  dtype=torch.long)