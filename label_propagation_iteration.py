import json
import numpy as np 
import os
import time
import torch

from compatibility import compatibility_from_text_features, compatibility_from_potts, compatibility_from_indicator
from iteration import label_prop
from potentials import unitary_potential_from_softmax
from utils import calculate_accuracy
from utils_sparsity import determine_sparsity_split_split
from utils_annotations import get_annotation_iterative, annotated_unitary_potential
from variance import custom_variance, variance_from_image, variance_from_features


EVALUATE = True
device = 'cuda'


variances_correspondance = {
    "custom": custom_variance,
    "image": variance_from_image, 
    "features": variance_from_features
}

compatibility_correspondance = {
    "potts": compatibility_from_potts,
    "text": compatibility_from_text_features,
    "indicator": compatibility_from_indicator
}


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
        "linear": args.linear
    }  
    with open(json_path, "w") as f:
        json.dump(json_file, f, indent = 6) 
        print(f"Json file saved to {json_path}")

    return


def inference_lab_prop(args, npz_path, unitary_potential, sparse, cossim=None, feat=None, data=None):
    # Var for pairwise potential
    get_variances = variances_correspondance[args.var]
    variances = get_variances(args, npz_path)

    # Pairwise potential and compatibility matrix
    if args.sparse_method != 'None': 
        from sparse_potentials import pairwise_potential_from_model_features_and_position, pairwise_potential_from_model_features, pairwise_potential_from_minus_model_features, pairwise_potential_from_model_features_ann
        potentials_correspondance = {
            "model_features_and_position": pairwise_potential_from_model_features_and_position, 
            "model_features": pairwise_potential_from_model_features,
            "minus_model_features": pairwise_potential_from_minus_model_features, 
            "model_features_ann": pairwise_potential_from_model_features_ann
        }

        img = None
        local_positions, nonlocal_positions, annotation_positions, _ = determine_sparsity_split_split(args, unitary_potential.size()[1], npz_path, img=img, features=feat, cossim=cossim)
        pairwise_potential = []
        
        if args.n_affinity[0] > 0:
            local_pairwise_potential = potentials_correspondance[args.pair_pot[0]](args, npz_path, local_positions, feat, variances, data).to(device)
            pairwise_potential.append(local_pairwise_potential)

        if args.n_affinity[1] > 0:
            i = 1
            if args.n_affinity[0] == 0: i=0
            nonlocal_pairwise_potential = potentials_correspondance[args.pair_pot[i]](args, npz_path, nonlocal_positions, feat, variances, data).to(device)
            pairwise_potential.append(nonlocal_pairwise_potential)

        if args.n_annotations > 0:
            annotation_pairwise_potential = potentials_correspondance[args.pair_pot[-1]](args, npz_path, annotation_positions, feat, variances, args.weight).to(device)
            pairwise_potential.append(annotation_pairwise_potential)

    else:
        from potentials import pairwise_potential_from_model_features,  \
                pairwise_potential_from_model_features_and_position
        
        potentials_correspondance = {
            "model_features": pairwise_potential_from_model_features,
            "model_features_and_position": pairwise_potential_from_model_features_and_position,
        }

        pairwise_potential = potentials_correspondance[args.pair_pot[0]](npz_path, variances, args.weight) 
        pairwise_potential = [torch.tensor(pairwise_potential, dtype=torch.float16, device=device)] 

    # Inference
    time0_inference = time.time()
    Q = label_prop(args, npz_path, unitary_potential, pairwise_potential, sparse) #, compatibility, variances, max_iters=args.n_iterations, cossim=cossim) # implement other propagation method
    time1_inference = time.time()
    inference_time = time1_inference-time0_inference
    print("Time to compute label propagation", inference_time)

    map_accuracy, map_balanced_accuracy = None, None
    if EVALUATE: 
        map_labels = torch.argmax(Q, dim=0)
        map_accuracy, map_balanced_accuracy = calculate_accuracy(npz_path, map_labels.cpu())
    new_probabilities = torch.clone(Q) # * args.temperature

    return map_accuracy, map_balanced_accuracy, new_probabilities 


def label_propagation(args, npz_path, sparse):
    data = np.load(npz_path)
    if args.dataset == 'esca':
        npz_name_ = f'{args.dataset}_{args.model}_cossim.npz' 
        npz_path_ = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model, npz_name_)
        cossim = torch.tensor(np.load(npz_path_)['cossim'], dtype=torch.int16, device=device)
    else:
        cossim = torch.tensor(data['cossim'], dtype=torch.int32, device=device)
    feat = torch.tensor(data['features'], dtype=torch.float32, device=device)
    probabilities = torch.tensor(data['probabilities'], dtype=torch.float16)
    args.width = data["width"]
    args.height = data["height"]
    probabilities = torch.tensor(data["probabilities"], dtype=torch.float16, device=device)
    labels = torch.tensor(data["labels"], device=device)
    
    weights = np.copy(args.weight)

    # Get unitary pot
    initial_unitary_potential = torch.tensor(unitary_potential_from_softmax(args, probabilities), dtype=torch.float16, device=device)
    unitary_potential = initial_unitary_potential.clone()
    initial_labels = torch.argmin(initial_unitary_potential, dim=0).cpu()
    
    if EVALUATE:
        initial_accuracy, initial_balanced_accuracy = calculate_accuracy(npz_path, initial_labels) # Evaluate initial_labels 

    annotation_accuracies, map_accuracies = [], []
    annotation_balanced_accuracies, map_balanced_accuracies = [], []
    all_it = []
    all_predicted_labels = []
    
    args.it = 0
    for it in range(args.ann_iterations):
        if args.n_annotations > 0:
            # Get annotation based on args.annotation_method if args.n_annotations > 0 
            if it == 0: 
                args.annotations, args.label_annotation = get_annotation_iterative(args, npz_path)
            else:
                args.annotations, args.label_annotation = get_annotation_iterative(args, npz_path, probabilities=new_probabilities.T)

        # At every n_annotation, repropagate from the initial potential /!\/!\/!\
        if args.ann_iterations > 1:
            unitary_potential = torch.clone(initial_unitary_potential)

        # Update unitary pot with annotations 
        if args.n_annotations > 0:
            unitary_potential = annotated_unitary_potential(args, labels, unitary_potential)

        if EVALUATE: 
            annotated_labels = torch.argmin(unitary_potential, dim=0).cpu()
            annotation_accuracy, annotation_balanced_accuracy = calculate_accuracy(npz_path, annotated_labels) # Evaluate annotated_labels 

        for lab_prop_it in range(args.N_PROP):
            args.it = it*args.N_PROP + lab_prop_it 
            all_it.append(args.it)

            # Propagate labels 
            map_accuracy, map_balanced_accuracy, new_probabilities = inference_lab_prop(args, npz_path, unitary_potential, sparse, cossim=cossim, feat=feat, data=data)
            new_probabilities = new_probabilities.cpu().numpy()
            map_accuracies.append(map_accuracy)
            map_balanced_accuracies.append(map_balanced_accuracy)

            annotation_accuracies.append(annotation_accuracy)
            annotation_balanced_accuracies.append(annotation_balanced_accuracy)

            # New unitary potential 
            unitary_potential_it = new_probabilities
            alpha = 0.5

            unitary_potential_it = np.array(unitary_potential_it, dtype=float)
            unitary_potential = np.array(unitary_potential.cpu(), dtype=float)

            unitary_potential = (1-alpha)*unitary_potential_it + alpha*unitary_potential 
            unitary_potential = torch.tensor(unitary_potential, dtype=torch.float16, device=device)

            if args.n_annotations > 0:
                unitary_potential = annotated_unitary_potential(args, labels, unitary_potential)
        
        if it % 1 == 0: 
            predicted_labels = np.argmax(new_probabilities, axis=0)
            all_predicted_labels.append(predicted_labels)

    args.weight = np.copy(weights)

    if EVALUATE: 
        n_labels = np.load(npz_path)["n_labels"]
        gt_labels = np.load(npz_path)["labels"]
        predicted_labels = np.argmax(new_probabilities, axis=0)
        all_predicted_labels.append(predicted_labels)
        for i in range(n_labels):
            print(f"Number of labels of class {i} gt, before and after: {gt_labels[gt_labels == i].shape[0]} >< {initial_labels[initial_labels == i].shape[0]} >< {predicted_labels[predicted_labels == i].shape[0]}")

        if args.n_annotations > 0:
            print(f"Label annotation = {args.label_annotation}")
            initial_unitary_potential_annotated = annotated_unitary_potential(args, labels, initial_unitary_potential)
            initial_annotated_labels = torch.argmin(initial_unitary_potential_annotated, dim=0).cpu()
            initial_annotated_accuracy, initial_annotated_balanced_accuracy = calculate_accuracy(npz_path, initial_annotated_labels) # Evaluate initial_labels 
            print(f"Initial annotated accuracy = {initial_annotated_accuracy}")
        print(f"Initial accuracy = {initial_accuracy}")
        print(f"Annotation accuracies = {annotation_accuracies}")
        print(f"Map accuracies = {map_accuracies}")
        print(f"Accuracies gain = {np.array(map_accuracies)-np.array(annotation_accuracies)}")
        print(f"args.it = {all_it}")

        accuracies_gain = np.array(map_accuracies)-np.array(annotation_accuracies)
        accuracies_balanced_gain = np.array(map_balanced_accuracies)-np.array(annotation_balanced_accuracies)
        if args.n_annotations > 0:
            results = [args.label_annotation, initial_accuracy, annotation_accuracies, map_accuracies, accuracies_gain, annotation_balanced_accuracies,  map_balanced_accuracies, accuracies_balanced_gain, initial_annotated_accuracy, initial_balanced_accuracy, initial_annotated_balanced_accuracy, all_predicted_labels]
        else:
            results = [np.array(-1), initial_accuracy, annotation_accuracies, map_accuracies, accuracies_gain, annotation_balanced_accuracies, map_balanced_accuracies, accuracies_balanced_gain, -1, initial_balanced_accuracy, -1, all_predicted_labels]
        save_results(args, npz_path, results)

    return 