import argparse
import json 
import numpy as np
import os
from PIL import Image
#from scipy.spatial.distance import dice 
from sklearn.metrics import balanced_accuracy_score, recall_score  #jaccard_score
import time 
import torch 
#import tracemalloc
from memory_profiler import profile

from compatibility import compatibility_from_text_features, compatibility_from_potts
from iteration import mean_field_iteration #, mean_field_iteration_sparse 
from potentials import unitary_potential_from_softmax, unitary_potential_from_softmax_and_annotation
from variance import custom_variance, variance_from_image, variance_from_features
from utils import dice, convert_to_sparse, determine_sparsity, annotated_unitary_potential, get_annotation
#from utils_patches import get_similar_patches, fill_sparse_matrix

device = 'cuda'

global SAVE 
SAVE = True

variances_correspondance = {
    "custom": custom_variance,
    "image": variance_from_image, 
    "features": variance_from_features
}

compatibility_correspondance = {
    "potts": compatibility_from_potts,
    "text": compatibility_from_text_features
}

unitary_correspondance = {
    "softmax": unitary_potential_from_softmax,
    "softmax_and_annotation": unitary_potential_from_softmax_and_annotation
}


def save_segmentation_mask(args, gt_labels, initial_labels, final_labels, npz_name):
    #height, width = args.patch_size @int(np.sqrt(np.shape(gt_labels)[0])), int(np.sqrt(np.shape(gt_labels)[0]))
    height, width = np.shape(gt_labels)[0], 1   

    # Create directory 
    results_dir = os.path.join(args.root_dir, 'results', args.dataset, args.model)
    if args.patch_size != 224:
        results_dir = os.path.join(args.root_dir, 'results', args.dataset, args.model)
    gt_file_name = os.path.join(results_dir, npz_name + '_gt_mask.png')
    first_file_name = os.path.join(results_dir, npz_name + '_first_mask.png')
    final_file_name = os.path.join(results_dir, npz_name + 'final_mask.png')
    
    # Ground truth 
    gt_mask = (gt_labels.reshape(width, height)*255).astype(np.uint8)
    Image.fromarray(gt_mask).save(gt_file_name)

    # First 
    first_mask = (initial_labels.reshape(width, height)*255).astype(np.uint8)
    Image.fromarray(first_mask).save(first_file_name)

    # Final 
    final_mask = (final_labels.reshape(width, height)*255).astype(np.uint8)
    Image.fromarray(final_mask).save(final_file_name)

    return f"Mask saved to {results_dir}"


def evaluate(args, final_labels, npz_path, info_to_save, annotated_labels=None, unitary=None):
    """
    Evaluate the performance of the model and save results.

    Args:
        final_labels (torch.Tensor): Predicted labels.
        npz_path (str): Path to the .npz file.

    Returns:
        None
    """
    final_labels = final_labels.cpu().numpy()

    # Create directory
    results_dir = os.path.join(args.root_dir, 'results', args.dataset, args.model)
    if not os.path.exists(results_dir): 
        os.makedirs(results_dir)

    weight = args.weight
    if isinstance(weight, list):
        weight = [str(w) for w in weight]
        weight = 'n'.join(weight)

    npz_name = npz_path.split('.npz')[0].split(os.path.sep)[-1]
    var = args.var
    if args.var == 'custom':
        var = [str(v) for v in args.custom_var]
        var = 'n'.join(var)
    json_name = f'{str(args.n_iterations)}_{str(weight)}_{var}_{args.pair_pot.replace("_", "")}_{args.compat}_{str(args.temperature)}_{args.sparse_method}_{str(args.n_affinity)}_{str(args.n_annotations)}_{str(npz_name)}.json'
    json_path = os.path.join(results_dir, json_name)
    
    # Load the .npz file and access the required keys
    data = np.load(npz_path)
    gt_labels = data["labels"].reshape((-1)) 
    probabilities = data["probabilities"]
    probabilities = np.exp(-unitary.cpu().numpy()).T
    n_samples = len(gt_labels)

    # Initial performance 
    n_labels = 12
    #initial_class_accuracy = np.zeros(n_labels)
    print("probabilities.T", probabilities.T.shape)
    initial_labels = np.argmax(probabilities.T, axis=0).reshape((-1)) ######!!!!! Enlever le .T
    initial_accuracy = initial_labels[initial_labels == gt_labels].shape[0] / n_samples # accuracy
    initial_balanced_acc = balanced_accuracy_score(gt_labels, initial_labels)
    initial_per_class_acc = recall_score(gt_labels, initial_labels, average=None, labels=list(range(n_labels)), zero_division=0.)
    initial_dice = dice(initial_labels, gt_labels) # dice
    #for c in np.unique(gt_labels): 
    #    initial_labels_c = initial_labels == c
    #    gt_labels_c = gt_labels == c
    #    initial_class_accuracy[c] = (initial_labels_c == gt_labels_c)/ gt_labels_c
    print(f'Initial performance (accuracy and balanced accuracy): {initial_accuracy}, {initial_balanced_acc} ({initial_labels[initial_labels == gt_labels].shape[0] })')

    # Final performance
    final_accuracy = final_labels[final_labels == gt_labels].shape[0] / n_samples # accuracy
    final_balanced_acc = balanced_accuracy_score(gt_labels, final_labels)
    final_per_class_acc = recall_score(gt_labels, final_labels, average=None, labels=list(range(n_labels)), zero_division=0.)
    final_dice = dice(final_labels, gt_labels) # dice
    print(f'Final performance (accuracy and balanced accuracy): {final_accuracy}, {final_balanced_acc} ({final_labels[final_labels == gt_labels].shape[0] })')

    # Annotation performance
    annotation_accuracy, annotation_balanced_acc = -1, -1
    if len(args.annotations) > 0:
        annotation_accuracy = annotated_labels[annotated_labels == gt_labels].shape[0] / n_samples # accuracy
        annotation_balanced_acc = balanced_accuracy_score(gt_labels, annotated_labels)
        print(f'Annotated performance (accuracy): {annotation_accuracy} ({annotated_labels[annotated_labels == gt_labels].shape[0]})')

    # Performance per class
    n_class = 12
    for i in range(n_class):
        print(f"Number of pixel of class {i}: before {len(initial_labels[initial_labels == i])} and after {len(final_labels[final_labels == i])} >< ground truth {len(gt_labels[gt_labels == i])}")

    # Json file
    json_file = {
        "initial_accuracy": str(initial_accuracy), 
        "final_accuracy": str(final_accuracy),
        "initial_balanced_accuracy": str(initial_balanced_acc),
        "final_balanced_accuracy": str(final_balanced_acc),
        "initial_accuracy_per_class": str(initial_per_class_acc),
        "final_accuracy_per_class": str(final_per_class_acc),
        "annotation_accuracy": str(annotation_accuracy),
        "annotation_accuracy_per_class": str(annotation_balanced_acc),
        "initial_dice": str(initial_dice), 
        "final_dice": str(final_dice),
        "pairwise_time": str(info_to_save[0]),
        "inference_time": str(info_to_save[1]), 
        "memory_usage": str(info_to_save[2]),
        "memory_peak": str(info_to_save[3])
    }  
    with open(json_path, "w") as f:
        json.dump(json_file, f, indent = 6) 
        print(f"Json file saved to {json_path}")

    if SAVE: 
        save_segmentation_mask(args, gt_labels, initial_labels, final_labels, npz_name)
    return 

@profile
def inference(args, npz_path):
    # Track memory usage
    #tracemalloc.start()

    # Unitary potential 
    print("--- Unitary potential computation ---")
    get_unitary_potential = unitary_correspondance[args.uni_pot]
    unitary_potential = torch.tensor(get_unitary_potential(args, npz_path), dtype=torch.float16, device=device)
    initial_unitary_potential = torch.clone(unitary_potential)
    print(f"unitary: {torch.mean(unitary_potential)} +/- {torch.std(unitary_potential)}")
    # Annotation 
    annotated_labels = None
    if len(args.annotations) > 0:
        unitary_potential = annotated_unitary_potential(args, npz_path, unitary_potential)
        annotated_labels = torch.argmin(unitary_potential, dim=0).cpu()
        
    # Pairwise potential 

    #cos_sim = np.load(npz_path)['cos_sim']
    #similar_patches = get_similar_patches(args, cos_sim)
    #image = np.load(npz_path)['images']

    get_variances = variances_correspondance[args.var]
    variances = get_variances(args, npz_path)

    print("--- Pairwise potential computation ---")
    get_pairwise_potential = potentials_correspondance[args.pair_pot]
    time0_pairwise_potential = time.time()
    if args.sparse_method != 'None': 
        img = None
        if args.sparse_method in ['nonlocal', 'nonlocalrandom']:
            img = np.load(npz_path)['images']
        all_positions, factors = determine_sparsity(args, unitary_potential.size()[1], npz_path, img=img)
        pairwise_potential = get_pairwise_potential(args, npz_path, all_positions, factors, variances, args.weight).to(device)

    else: 
        pairwise_potential = get_pairwise_potential(npz_path, variances, args.weight)
        if args.zero_method != 'None':
            img = None
            if args.sparse_method in ['nonlocal', 'nonlocalrandom']:
                img = np.load(npz_path)['images']
            pairwise_potential = convert_to_sparse(args, pairwise_potential, img=img).to(device)
        else:
            pairwise_potential = torch.tensor(pairwise_potential, dtype=torch.float16, device=device)
    
    #pairwise_potential = fill_sparse_matrix(torch.Tensor(image), variances, args.weight, similar_patches, sim_func=get_pairwise_potential)
    
    time1_pairwise_potential = time.time()
    pairwise_time = time1_pairwise_potential-time0_pairwise_potential
    print(f'Time to compute pairwise potential: {pairwise_time}s')
    get_compatibility = compatibility_correspondance[args.compat]
    compatibility = torch.tensor(get_compatibility(args, npz_path),dtype=torch.float16, device=device)
    #compatibility = torch.tensor([[0.2, 0.8, 0.8, 0.8], [0.8, 0.2, 0.8, 0.8], [0.8, 0.8, 0.2, 0.8], [0.8, 0.8, 0.8, 0.2]], dtype=torch.float16, device=device)
    #compatibility = torch.tensor([[0.6, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0.15, 1], [1, 1, 1, 0]], dtype=torch.float16, device=device)
    #compatibility = torch.tensor([[0.1, 1, 1, 1], [1, 0.3, 1, 1], [1, 1, 0.2, 1], [1, 1, 1, 0.1]], dtype=torch.float16, device=device)

    # Automatic weight 
    if args.weight[0] == -1:
        args.weight[0] = torch.mean(unitary_potential) / (args.weight[1])

    # Inference
    print("--- Start inference ---")
    time0_inference = time.time()
    Q = mean_field_iteration(args, npz_path, unitary_potential, pairwise_potential, compatibility, variances, max_iters=args.n_iterations)
    time1_inference = time.time()
    inference_time = time1_inference-time0_inference
    #print(f'Time to infer: {inference_time}s')

    map = torch.argmax(Q, dim=0)
    a = torch.argmin(unitary_potential, dim=0)
    print('Number of pixels modified', map[map != a].shape)

    # Evaluation 
    print("--- Start evaluation ---")
    #memory_usage, memory_peak = tracemalloc.get_traced_memory()
    memory_peak = torch.cuda.max_memory_allocated()
    memory_usage = torch.cuda.max_memory_reserved() #torch.cuda.memory_allocated()
    info_to_save = [pairwise_time, inference_time, memory_usage, memory_peak]
    #print("Memory usage and peak memory usage (in bytes)", memory_usage, memory_peak)
    #print("Memory summary", torch.cuda.memory_summary())
    #tracemalloc.stop()
    evaluate(args, map, npz_path, info_to_save, annotated_labels, unitary=initial_unitary_potential)

    del unitary_potential, pairwise_potential, map

    return 


def parser():
    pair_pot_choices = ['image_features', 'model_features', 
                        'image_and_position', 'image', 
                        'hsv_image_and_position', 'binary_mask', 
                        'hsl_image_and_position', 
                        'cielab_image_and_position', 'edges',
                        'model_features_and_position', 
                        'cos_sim', 'patch', 'prob']
    parser = argparse.ArgumentParser()
    # Setup 
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--dataset', type=str, default='skincancer2', choices=['sicap_mil', 'monuseg', 'skincancer', 'panuke', 'skincancer2', 'lc_lung', 'lc_colon', 'nct'])
    parser.add_argument('--model', type=str, default='nnunet', choices=['clip', 'quilt', 'conch', 'plip', 'nnunet', 'nnunet_patches', 'nnunet_bigpatches', 'plipanduni2'])
    # CRF
    parser.add_argument('--n_iterations', type=int, default=25)
    parser.add_argument('--weight', type=float, nargs='+', default=1)
    parser.add_argument('--var', type=str, default='image', choices=['custom', 'image', 'features'])
    parser.add_argument('--custom_var', type=float, nargs='+', default=None)
    parser.add_argument('--uni_pot', type=str, default='softmax', choices=['softmax','softmax_and_annotation'])
    parser.add_argument('--pair_pot', type=str, default='image_and_position', choices=pair_pot_choices)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--compat', type=str, default='text', choices=['potts', 'text']) 
    # Patches
    parser.add_argument('--sim_patches', type=int, default=1)
    parser.add_argument('--patch_size', type=int, nargs='+', default=112)
    # Sparsity
    parser.add_argument('--sparse_method', type=str, default='None', choices=['random', 'threshold', 'neighbor', 'nonlocal', 'localneighbor', 'nonlocalrandom', 'randomsparse', 'oracle', 'zsprob', 'cossim'])
    parser.add_argument('--zero_method', type=str, default='None', choices=['random', 'threshold', 'neighbor', 'nonlocal', 'localneighbor', 'nonlocalrandom', 'randomsparse'])
    parser.add_argument('--threshold', type=float, default=1e-1)
    parser.add_argument('--n_affinity', type=int, nargs='+')
    # Few shot 
    parser.add_argument('--annotations', type=int, nargs='+', default=[])
    parser.add_argument('--n_annotations', type=int, default=0)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parser()
    
    if args.sparse_method != 'None':
        from sparse_potentials import pairwise_potential_from_img_and_position, pairwise_potential_from_model_features_and_position, pairwise_potential_from_prob, pairwise_potential_from_model_features
        print("Sparse method not None")
        potentials_correspondance = {
            "image_and_position": pairwise_potential_from_img_and_position,
            "model_features_and_position": pairwise_potential_from_model_features_and_position, 
            "prob": pairwise_potential_from_prob, 
            "model_features": pairwise_potential_from_model_features
        }
    else:
        from potentials import pairwise_potential_from_model_features, \
                pairwise_potential_from_img_and_position, pairwise_potential_from_img, \
                pairwise_potential_from_hsv_image_and_position, pairwise_potential_from_binary_mask, \
                pairwise_potential_from_hsl_image_and_position, pairwise_potential_from_cielab_image_and_position, \
                pairwise_potential_from_edges, \
                pairwise_potential_from_model_features_and_position, pairwise_potential_from_img_features_and_cossim, \
                pairwise_potential_from_img_patches 
        
        potentials_correspondance = {
            "model_features": pairwise_potential_from_model_features,
            "image_and_position": pairwise_potential_from_img_and_position,
            "image": pairwise_potential_from_img,
            "hsv_image_and_position": pairwise_potential_from_hsv_image_and_position,
            "binary_mask": pairwise_potential_from_binary_mask, 
            "hsl_image_and_position": pairwise_potential_from_hsl_image_and_position, 
            "cielab_image_and_position": pairwise_potential_from_cielab_image_and_position,
            "edges": pairwise_potential_from_edges,
            "model_features_and_position": pairwise_potential_from_model_features_and_position,
            "cos_sim": pairwise_potential_from_img_features_and_cossim,
            "patch": pairwise_potential_from_img_patches
        }

    print(f"Start for sparse_method={args.sparse_method} and n_affinity={args.n_affinity}")
    if args.model in ['nnunet', 'nnunet_patches']:
        args.patch_size = args.patch_size[0]
        data_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
        npz_paths = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        if args.patch_size != 224:
            npz_paths = [f for f in npz_paths if str(args.patch_size) in f]
        else:
            npz_paths = [f for f in npz_paths if not any(substr in f for substr in ['50', '300', '400', '500', '1000'])]
        for npz_path in npz_paths[:1]:
            npz_path = os.path.join(data_dir, npz_path)
            torch.cuda.memory._record_memory_history()
            inference(args, npz_path)
            torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    elif args.model == 'nnunet_bigpatches':
        data_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model, str(args.patch_size))
        npz_paths = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        for npz_path in npz_paths:
            npz_path = os.path.join(data_dir, npz_path)
            torch.cuda.memory._record_memory_history()
            inference(args, npz_path)
            torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    elif args.model in ['plip', 'conch', 'clip', 'quilt', 'plipanduni2'] and args.dataset == 'skincancer2':
        data_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
        npz_paths = [f for f in os.listdir(data_dir) if f.endswith(".npz")] 
        w0 = args.weight[0]
        w1 = args.weight[1]
        for npz_path in npz_paths:
            npz_path = os.path.join(data_dir, npz_path)
            if args.n_annotations > 0:
                args.annotations = get_annotation(args, npz_path)
            #if '_0.npz' in npz_path: 
            args.weight[0] = w0
            args.weight[1] = w1
            inference(args, npz_path)
            
    else:
        args.patch_size = args.patch_size[0]
        data_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
        npz_path = os.path.join(data_dir, f'{args.dataset}_{args.model}.npz')
        if args.n_annotations > 0:
                args.annotations = get_annotation(args, npz_path)
        inference(args, npz_path)
