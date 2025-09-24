import json
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score, recall_score

device = 'cuda'


def save_npz(labeled_masks):
    npz_path = 'labeled_masks.npz'
    np.savez(npz_path, labeled_masks)
    return npz_path


def calculate_accuracy(npz_path, predicted_labels):
    gt_labels = np.load(npz_path)["labels"]

    n_samples = len(gt_labels)
    n_labels = len(np.unique(n_samples))

    predicted_accuracy = predicted_labels[predicted_labels == gt_labels].shape[0] / n_samples
    predicted_balanced_acc = balanced_accuracy_score(gt_labels, predicted_labels)
    predicted_per_class_acc = recall_score(gt_labels, predicted_labels, average=None, labels=list(range(n_labels)), zero_division=0.)

    return predicted_accuracy, predicted_balanced_acc


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


# !!! From StatA github https://github.com/MaxZanella/StatA/blob/main/solvers/StatA.py#L148 !!! 
def update_beta(z, alpha, soft=False):
    if soft:
        sum_z = np.sum(z, axis=0)
        beta = sum_z / (alpha + sum_z)
    else: 
        predicted_classes = np.argmax(z, axis=1)
        sum_z = np.bincount(predicted_classes, minlength=z.shape[1])  
        beta = sum_z / (alpha + sum_z + 1e-12)
    return beta      


def save_results(args, npz_path, results):
    # Create directory
    results_dir = os.path.join(args.root_dir, 'results', args.dataset, args.model)
    if not os.path.exists(results_dir): 
        os.makedirs(results_dir)

    weights = args.weight
    weight = [str(w) for w in weights]
    weight = ','.join(weight)   

    npz_name = npz_path.split('.npz')[0].split(os.path.sep)[-1]

    var = args.var
    if args.var == 'custom':
        var = [str(v) for v in args.custom_var]
        var = ','.join(var)

    pp = args.pair_pot.copy()
    for i, p in enumerate(pp): 
        pp[i] = p.replace("_", "")
    pp = ','.join(pp)

    compat = args.compat.copy()
    for i, c in enumerate(compat): 
        compat[i] = c.replace("_", "")
    compat = ','.join(compat)

    json_name = f'{args.N_PROP}_{args.N_UP}_{args.ann_iterations}_{str(args.n_iterations)}_{str(weight)}_{var}_{pp}_{compat}_{str(args.temperature)}_{args.sparse_method}_{str(args.n_affinity)}_{str(args.n_annotations)}_{str(args.n_class_not_annotated)}_{str(args.seed)}_{args.annotation_method}_{args.linear}_{str(npz_name)}.json'
    json_path = os.path.join(results_dir, json_name)

    results[11] = [arr.tolist() for arr in results[11]]
    # Json file
    json_file = {
        "label_annotation": results[0].tolist(),
        "initial_accuracy": results[1],
        "initial_annotation_accuracy": results[8],
        "initial_balanced_accuracy": results[9],
        "initial_annotated_balanced_accuracy": results[10],
        "annotation_accuracies": results[2],
        "map_accuracies": results[3],
        "accuracies_gain": results[4].tolist(),
        "annotation_balanced_accuracies": results[5],
        "map_balanced_accuracies": results[6],
        "accuracies_balanced_gain": results[7].tolist(),
        "ann_iterations": args.ann_iterations,
        "N_PROP": args.N_PROP,
        "N_UP": args.N_UP,
        "annotation_method": args.annotation_method,
        "n_annotations": args.n_annotations,
        "sparse_method": args.sparse_method,
        "n_affinity": str(args.n_affinity),
        "n_iterations": args.n_iterations, 
        "weight": weight,
        "var": var,
        "pp": pp,
        "compat": compat, 
        "temperature": args.temperature,
        "seed": args.seed,
        "final_label": results[11],
        "linear": args.linear, 
        "annotations": args.annotations
    }  
    with open(json_path, "w") as f:
        json.dump(json_file, f, indent = 6) 
        print(f"Json file saved to {json_path}")

    return


def save_results_gif(args, npz_path, results):
    # Create directory
    results_dir = os.path.join(args.root_dir, 'results', args.dataset, args.model)
    if not os.path.exists(results_dir): 
        os.makedirs(results_dir)

    weights = args.weight
    weight = [str(w) for w in weights]
    weight = ','.join(weight)   

    npz_name = npz_path.split('.npz')[0].split(os.path.sep)[-1]

    var = args.var
    if args.var == 'custom':
        var = [str(v) for v in args.custom_var]
        var = ','.join(var)

    pp = args.pair_pot.copy()
    for i, p in enumerate(pp): 
        pp[i] = p.replace("_", "")
    pp = ','.join(pp)

    compat = args.compat.copy()
    for i, c in enumerate(compat): 
        compat[i] = c.replace("_", "")
    compat = ','.join(compat)

    json_name = f'gif_{args.N_PROP}_{args.N_UP}_{args.ann_iterations}_{str(args.n_iterations)}_{str(weight)}_{var}_{pp}_{compat}_{str(args.temperature)}_{args.sparse_method}_{str(args.n_affinity)}_{str(args.n_annotations)}_{str(args.n_class_not_annotated)}_{str(args.seed)}_{args.annotation_method}_{args.linear}_{str(npz_name)}.json'
    json_path = os.path.join(results_dir, json_name)

    results = [arr.tolist() for arr in results]
    # Json file
    json_file = {
        "ann_iterations": args.ann_iterations,
        "N_PROP": args.N_PROP,
        "N_UP": args.N_UP,
        "annotation_method": args.annotation_method,
        "n_annotations": args.n_annotations,
        "sparse_method": args.sparse_method,
        "n_affinity": str(args.n_affinity),
        "n_iterations": args.n_iterations, 
        "weight": weight,
        "var": var,
        "pp": pp,
        "compat": compat, 
        "temperature": args.temperature,
        "seed": args.seed,
        "final_label": results,
        "linear": args.linear, 
        "annotations": args.annotations
    }  
    with open(json_path, "w") as f:
        json.dump(json_file, f, indent = 6) 
        print(f"Json file saved to {json_path}")

    return