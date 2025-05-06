import argparse
import json 
import numpy as np
import os
from PIL import Image
#from scipy.spatial.distance import dice 
#from sklearn.metrics import jaccard_score
import time 
import torch 
import tracemalloc

from compatibility import compatibility_from_text_features, compatibility_from_potts
from iteration import mean_field_iteration 
from potentials import pairwise_potential_from_model_features, unitary_potential_from_softmax, \
        pairwise_potential_from_img_and_position, pairwise_potential_from_img, \
        pairwise_potential_from_hsv_image_and_position, pairwise_potential_from_binary_mask, \
        pairwise_potential_from_hsl_image_and_position, pairwise_potential_from_cielab_image_and_position, \
        pairwise_potential_from_edges, \
        pairwise_potential_from_model_features_and_position, pairwise_potential_from_img_features_and_cossim
from variance import custom_variance, variance_from_image, variance_from_features
from utils import dice

device = 'cuda'

global SAVE 
SAVE = False

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
    "cos_sim": pairwise_potential_from_img_features_and_cossim
}

variances_correspondance = {
    "custom": custom_variance,
    "image": variance_from_image, 
    "features": variance_from_features
}

compatibility_correspondance = {
    "potts": compatibility_from_potts,
    "text": compatibility_from_text_features
}


def save_segmentation_mask(args, gt_labels, initial_labels, final_labels, npz_name):
    width, height = int(np.sqrt(np.shape(gt_labels)[0])), int(np.sqrt(np.shape(gt_labels)[0]))

    # Create directory 
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


def evaluate(args, final_labels, npz_path, info_to_save):
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
    json_name = f'{str(args.n_iterations)}_{str(weight)}_{var}_{args.pair_pot.replace("_", "")}_{args.compat}_{str(args.temperature)}_{str(npz_name)}.json'
    json_path = os.path.join(results_dir, json_name)
    
    # Load the .npz file and access the required keys
    data = np.load(npz_path)
    gt_labels = data["labels"].reshape((-1)) 
    probabilities = data["probabilities"]
    n_samples = len(gt_labels)

    # Initial performance 
    initial_labels = np.argmax(probabilities, axis=0).reshape((-1)) ######!!!!! Enlever le .T
    initial_accuracy = initial_labels[initial_labels == gt_labels].shape[0] / n_samples # accuracy
    initial_dice = dice(initial_labels, gt_labels) # dice
    print(f'Initial performance (accuracy, dice): {initial_accuracy}, {initial_dice}')

    # Final performance
    final_accuracy = final_labels[final_labels == gt_labels].shape[0] / n_samples # accuracy
    final_dice = dice(final_labels, gt_labels) # dice
    print(f'Final performance (accuracy, dice): {final_accuracy}, {final_dice}')

    # Json file
    json_file = {
        "initial_accuracy": str(initial_accuracy), 
        "final_accuracy": str(final_accuracy),
        "initial_dice": str(initial_dice), 
        "final_dice": str(final_dice),
        "pairwise_time": str(info_to_save[0]),
        "inference_time": str(info_to_save[1]),
        "memory_usage": str(info_to_save[2]),
        "memory_peak": str(info_to_save[3])
    }  
    with open(json_path, "w") as f:
        json.dump(json_file, f, indent = 6) 

    if SAVE: 
        save_segmentation_mask(args, gt_labels, initial_labels, final_labels, npz_name)

    return 


def inference(args, npz_path):
    # Unitary potential 
    print("--- Unitary potential computation ---")
    #unitary_potential = torch.Tensor(unitary_potential_from_softmax(npz_path), device=device)
    unitary_potential = torch.tensor(unitary_potential_from_softmax(npz_path), dtype=torch.float16, device=device)

    # Pairwise potential 
    get_variances = variances_correspondance[args.var]
    variances = get_variances(args, npz_path)
    print("--- Pairwise potential computation ---")
    get_pairwise_potential = potentials_correspondance[args.pair_pot]
    time0_pairwise_potential = time.time()
    pairwise_potential = torch.tensor(get_pairwise_potential(npz_path, variances, args.weight), dtype=torch.float16, device=device)
    time1_pairwise_potential = time.time()
    pairwise_time = time1_pairwise_potential-time0_pairwise_potential
    print(f'Time to compute pairwise potential: {time1_pairwise_potential-time0_pairwise_potential}s')
    get_compatibility = compatibility_correspondance[args.compat]
    compatibility = torch.tensor(get_compatibility(args, npz_path),dtype=torch.float16, device=device)

    # Inference 
    print("--- Start inference ---")
    time0_inference = time.time()
    Q = mean_field_iteration(args, unitary_potential, pairwise_potential, compatibility, max_iters=args.n_iterations)
    time1_inference = time.time()
    inference_time = time1_inference-time0_inference
    print(f'Time to infer: {time1_inference-time0_inference}s')
    map = torch.argmax(Q, dim=0)

    # Evaluation 
    print("--- Start evaluation ---")
    memory_peak = torch.cuda.max_memory_allocated()
    memory_usage = torch.cuda.memory_allocated()
    #memory_usage, memory_peak = tracemalloc.get_traced_memory()
    info_to_save = [pairwise_time, inference_time, memory_usage, memory_peak]
    print("Memory usage and peak memory usage (in bytes)", memory_usage, memory_peak)
    #tracemalloc.stop()
    evaluate(args, map, npz_path, info_to_save)

    del unitary_potential, pairwise_potential, map

    return 


def parser():
    pair_pot_choices = ['image_features', 'model_features', 
                        'image_and_position', 'image', 
                        'hsv_image_and_position', 'binary_mask', 
                        'hsl_image_and_position', 
                        'cielab_image_and_position', 'edges',
                        'model_features_and_position', 
                        'cos_sim']
    parser = argparse.ArgumentParser()
    # Setup 
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--dataset', type=str, default='sicap_mil', choices=['sicap_mil', 'monuseg', 'skincancer', 'panuke', 'skincancer2'])
    parser.add_argument('--model', type=str, default='clip', choices=['clip', 'quilt', 'conch', 'plip', 'nnunet', 'nnunet_bigpatches'])
    # CRF
    parser.add_argument('--n_iterations', type=int, default=25)
    parser.add_argument('--weight', type=float, nargs='+', default=1)
    parser.add_argument('--var', type=str, default='image', choices=['custom', 'image', 'features'])
    parser.add_argument('--custom_var', type=float, nargs='+', default=None)
    parser.add_argument('--pair_pot', type=str, default='model_features', choices=pair_pot_choices)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--compat', type=str, default='text', choices=['potts', 'text']) 
     # Patches
    parser.add_argument('--sim_patches', type=int, default=None) 
    parser.add_argument('--patch_size', type=int, default=224) 
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parser()
   
    if args.model in ['nnunet', 'nnunet_bigpatches']:
        print(f"Use of {args.model}")
        data_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
        npz_paths = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        for npz_path in npz_paths[:2]:
            npz_path = os.path.join(data_dir, npz_path)
            inference(args, npz_path)

    else:
        data_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
        npz_path = os.path.join(data_dir, f'{args.dataset}_{args.model}.npz')
        inference(args, npz_path)
