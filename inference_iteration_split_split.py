import argparse
import numpy as np 
import os
import time
import torch

from compatibility import compatibility_from_text_features, compatibility_from_potts, compatibility_from_indicator
from iteration import mean_field_iteration_split
from label_propagation_iteration import label_propagation
from potentials import unitary_potential_from_softmax, unitary_potential_from_softmax_and_annotation
from train_linear import train_linear
from utils_sparsity import determine_sparsity_split_split
from utils import calculate_accuracy, save_results_gif, save_results
from utils_annotations import get_annotation_iterative, unitary_potential_from_softmax_unlabeled_and_annotation, annotated_unitary_potential
from variance import custom_variance, variance_from_image, variance_from_features


EVALUATE = True
GIF = False
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


def inference(args, npz_path, unitary_potential, cossim=None, feat=None, data=None):
    # Var for pairwise potential
    get_variances = variances_correspondance[args.var]
    variances = get_variances(args, npz_path)

    # Pairwise potential and compatibility matrix
    if args.sparse_method == 'None': 
        pairwise_potential = potentials_correspondance[args.pair_pot[0]](npz_path, variances, args.weight) 
        pairwise_potential = [torch.tensor(pairwise_potential, dtype=torch.float16, device=device)] 
    else:
        img = None
        if args.sparse_method in ['nonlocal', 'nonlocalrandom']:
            img = np.load(npz_path)['images']

        local_positions, nonlocal_positions, annotation_positions, _ = determine_sparsity_split_split(args, unitary_potential.size()[1], npz_path, img=img, features=feat, cossim=cossim)

        pairwise_potential, compatibility = [], []
        if args.n_affinity[0] > 0:
            local_pairwise_potential = potentials_correspondance[args.pair_pot[0]](args, npz_path, local_positions, feat, variances, data).to(device)
            pairwise_potential.append(local_pairwise_potential)

            local_compatibility =  torch.tensor(compatibility_correspondance[args.compat[0]](args, npz_path), dtype=torch.float16, device=device)
            compatibility.append(local_compatibility)

        if args.n_affinity[1] > 0:
            i = 1
            if args.n_affinity[0] == 0: i=0
            nonlocal_pairwise_potential = potentials_correspondance[args.pair_pot[i]](args, npz_path, nonlocal_positions, feat, variances, data).to(device)
            pairwise_potential.append(nonlocal_pairwise_potential)

            nonlocal_compatibility =  torch.tensor(compatibility_correspondance[args.compat[i]](args, npz_path), dtype=torch.float16, device=device)
            compatibility.append(nonlocal_compatibility)

        if args.n_annotations > 0:
            annotation_pairwise_potential = potentials_correspondance[args.pair_pot[-1]](args, npz_path, annotation_positions, feat, variances, args.weight).to(device)
            pairwise_potential.append(annotation_pairwise_potential)

            annotation_compatibility = torch.tensor(compatibility_correspondance[args.compat[-1]](args, npz_path), dtype=torch.float16, device=device)
            compatibility.append(annotation_compatibility)

    # Inference
    time0_inference = time.time()
    Q = mean_field_iteration_split(args, npz_path, unitary_potential, pairwise_potential, compatibility, variances, max_iters=args.n_iterations, cossim=None, data=data)
    time1_inference = time.time()
    inference_time = time1_inference-time0_inference
    print("Time to compute one iterative message passing", inference_time)

    map_accuracy, map_balanced_accuracy = None, None
    if EVALUATE: 
        map_labels = torch.argmax(Q, dim=0)
        map_accuracy, map_balanced_accuracy = calculate_accuracy(npz_path, map_labels.cpu())
    new_probabilities = torch.clone(Q) 

    return map_accuracy, map_balanced_accuracy, new_probabilities 


def inference_iteration(args, npz_path):
    weights = np.copy(args.weight)

    data = np.load(npz_path)
    if args.dataset == 'esca':
        npz_name_ = f'{args.dataset}_{args.model}_cossim.npz' 
        npz_path_ = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model, npz_name_)
        cossim = torch.tensor(np.load(npz_path_)['cossim'], dtype=torch.int16, device=device)
    else:
        cossim = torch.tensor(data['cossim'], dtype=torch.int32, device=device)
    feat = torch.tensor(data['features'], dtype=torch.float32, device=device)
    args.width = data["width"]
    args.height = data["height"]
    probabilities = torch.tensor(data["probabilities"], dtype=torch.float16, device=device)
    labels = torch.tensor(data["labels"], device=device)

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

    if GIF:
        all_predicted_labels_gif = []
    
    args.it = 0
    for it in range(args.ann_iterations):
        args.ann_it = it
        if args.n_annotations > 0:
            # Get annotation based on args.annotation_method if args.n_annotations > 0 
            if it == 0: 
                start = time.time()
                args.annotations, args.label_annotation = get_annotation_iterative(args, npz_path)
                #print("Time to get annnotation", time.time()-start)
            else:
                args.annotations, args.label_annotation = get_annotation_iterative(args, npz_path, probabilities=new_probabilities.T.cpu().numpy())

        # Train linear classifier to update unitary pot
        if args.linear == 'True': 
            new_probs = train_linear(args)
            softmax = torch.nn.Softmax(dim=1)
            probabilities = softmax(new_probs) 
            unitary_potential = -torch.log(probabilities)
            unitary_potential = torch.tensor(unitary_potential, dtype=torch.float16, device=device).t()

        # Compute prototype to update unitary pot
        if args.N_UP > it: 
            n_unitary_update = 1
            for unitary_it in range(n_unitary_update):
                if it == 0:
                    unitary_potential, updated_text_features = unitary_potential_from_softmax_and_annotation(args, npz_path)
                    unitary_potential = torch.tensor(unitary_potential, dtype=torch.float16, device=device)
                else:
                    unitary_potential, updated_text_features = unitary_potential_from_softmax_unlabeled_and_annotation(args, npz_path, probabilities=new_probabilities.T, text_features=updated_text_features) #/!\/!\/!\
                    unitary_potential = torch.tensor(unitary_potential, dtype=torch.float16, device=device)

        # At every n_annotation, repropagate from the initial potential /!\/!\/!\
        if args.ann_iterations > 1:
            unitary_potential = torch.clone(initial_unitary_potential) 

        # Update unitary pot with annotations 
        if args.n_annotations > 0:
            unitary_potential = annotated_unitary_potential(args, labels, unitary_potential)

        if EVALUATE: 
            annotated_labels = torch.argmin(unitary_potential, dim=0).cpu()
            annotation_accuracy, annotation_balanced_accuracy = calculate_accuracy(npz_path, annotated_labels) # Evaluate annotated_labels 

        if args.PROP: 
            for lab_prop_it in range(args.N_PROP):
                args.it = it*args.N_PROP + lab_prop_it 
                all_it.append(args.it)

                # Propagate labels 
                map_accuracy, map_balanced_accuracy, new_probabilities = inference(args, npz_path, unitary_potential, cossim=cossim, feat=feat, data=data)
                map_accuracies.append(map_accuracy)
                map_balanced_accuracies.append(map_balanced_accuracy)

                if EVALUATE:
                    annotation_accuracies.append(annotation_accuracy)
                    annotation_balanced_accuracies.append(annotation_balanced_accuracy)

                # New unitary potential 
                unitary_potential_it = -(torch.log(new_probabilities))
                alpha = 0.5
                unitary_potential = (1-alpha)*unitary_potential_it + alpha*unitary_potential

                del unitary_potential_it

                if args.n_annotations > 0:
                    unitary_potential = annotated_unitary_potential(args, labels, unitary_potential)

                torch.cuda.empty_cache()

                if GIF: 
                    predicted_labels = torch.argmax(new_probabilities, dim=0)
                    all_predicted_labels_gif.append(predicted_labels.cpu().numpy())
   
        else:
            if it == args.ann_iterations - 1:   #### est-ce que c'est ann_iterations ici??? Verif la diff entre n_annotation et ann_iteration?
                # Propagate labels 
                map_accuracy, map_balanced_accuracy, new_probabilities = inference(args, npz_path, unitary_potential)
                new_probabilities = new_probabilities.cpu().numpy()
                map_accuracies.append(map_accuracy)
                map_balanced_accuracies.append(map_balanced_accuracy)

                # New unitary potential 
                unitary_potential = -(np.log(new_probabilities))
                unitary_potential = torch.tensor(unitary_potential, dtype=torch.float16, device=device)
            
            else:
                new_probabilities = -unitary_potential.cpu().numpy()
                map_accuracies.append(annotation_accuracy)
                map_balanced_accuracies.append(annotation_balanced_accuracy)

            if args.n_annotations > 0:
                unitary_potential = annotated_unitary_potential(args, npz_path, unitary_potential)

        if EVALUATE:
            if it % 1 == 0: 
                predicted_labels = torch.argmax(new_probabilities, dim=0)
                all_predicted_labels.append(predicted_labels)

    args.weight = np.copy(weights)

    if EVALUATE:
        n_labels = np.load(npz_path)["n_labels"]
        gt_labels = np.load(npz_path)["labels"]
        predicted_labels = torch.argmax(new_probabilities, dim=0)
        all_predicted_labels.append(predicted_labels.cpu().numpy())
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

    if GIF:
        save_results_gif(args, npz_path, all_predicted_labels_gif)

    return 


def parser():
    pair_pot_choices = ['model_features',
                        'model_features_and_position', 
                        'minus_model_features', 
                        'minus_model_features_ann']

    parser = argparse.ArgumentParser()
    # Setup 
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--dataset', type=str, default='skincancer2', choices=['sicap_mil', 'monuseg', 'skincancer', 'panuke', 'skincancer2', 'lc_lung', 'lc_colon', 'nct', 'bracs', 'tcga-ut', 'bach', 'esca', 'tcga_uniform', 'bach_wsi'])
    parser.add_argument('--model', type=str, default='nnunet', choices=['clip', 'quilt', 'conch', 'plip', 'nnunet', 'nnunet_patches', 'nnunet_bigpatches', 'plipanduni2', 'conchanduni2', 'conchandgigapath', 'conchandoptimus1'])
    parser.add_argument('--n_patient', type=int, default=-1)
    # CRF
    parser.add_argument('--n_iterations', type=int, default=25)
    parser.add_argument('--weight', type=float, nargs='+', default=1)
    parser.add_argument('--var', type=str, default='image', choices=['custom', 'image', 'features'])
    parser.add_argument('--custom_var', type=float, nargs='+', default=None)
    parser.add_argument('--uni_pot', type=str, default='softmax', choices=['softmax','softmax_and_annotation', 'softmax_unlabeled_and_annotation'])
    parser.add_argument('--pair_pot', type=str, nargs='+', default='image_and_position', choices=pair_pot_choices)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--compat', type=str, nargs='+', default='text', choices=['potts', 'text', 'indicator']) 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=-1)
    parser.add_argument('--height', type=int, default=-1)
    # Patches
    parser.add_argument('--sim_patches', type=int, default=1)
    parser.add_argument('--patch_size', type=int, nargs='+', default=112)
    # Sparsity
    parser.add_argument('--sparse_method', type=str, default='None', choices=['None', 'random', 'threshold', 'neighbor', 'nonlocal', 'localneighbor', 'nonlocalrandom', 'randomsparse', 'oracle', 'zsprob', 'cossim'])
    parser.add_argument('--threshold', type=float, default=1e-1)
    parser.add_argument('--n_affinity', type=int, nargs='+')
    # Few shot 
    parser.add_argument('--annotation_method', type=str, default='random', choices=['random', 'entropy', 'error', 'oracle', 'least_confident', 'margin', 'diverseerror', 'circle'])
    parser.add_argument('--annotations', type=int, nargs='+', default=[])
    parser.add_argument('--n_annotations', type=int, default=0)
    parser.add_argument('--n_class_not_annotated', type=int, default=0)
    parser.add_argument('--label_annotation', type=int, nargs='+', default=[])
    parser.add_argument('--classes_not_annotated', type=int, nargs='+', default=[])
    parser.add_argument('--ann_iterations', type=int, default=10)
    parser.add_argument('--ann_it', type=int, default=0)
    # Update unitary
    parser.add_argument('--beta', type=bool, default=False)
    # Exp 
    parser.add_argument('--PROP', type=bool)
    parser.add_argument('--N_UP', type=int, default=-1)
    parser.add_argument('--N_PROP', type=int, default=1)
    parser.add_argument('--it', type=int, default=0)
    parser.add_argument('--crf_it', type=int, default=0)
    parser.add_argument('--weight_initial', type=float, nargs='+', default=None)
    parser.add_argument('--annotated_row_positions', type=int, nargs='+', default=[])
    parser.add_argument('--linear', type=str, default='False')
    # Linear model
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_iters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser()

    if args.sparse_method != 'None':
        from sparse_potentials import pairwise_potential_from_model_features_and_position, pairwise_potential_from_model_features, pairwise_potential_from_minus_model_features, pairwise_potential_from_model_features_ann
        potentials_correspondance = {
            "model_features_and_position": pairwise_potential_from_model_features_and_position, 
            "model_features": pairwise_potential_from_model_features,
            "minus_model_features": pairwise_potential_from_minus_model_features, 
            "model_features_ann": pairwise_potential_from_model_features_ann
        }
    else:
        from potentials import pairwise_potential_from_model_features,  \
                pairwise_potential_from_model_features_and_position
        
        potentials_correspondance = {
            "model_features": pairwise_potential_from_model_features,
            "model_features_and_position": pairwise_potential_from_model_features_and_position,
        }

    args.weight_initial = np.copy(args.weight)

    if args.dataset in ['skincancer2', 'bach_wsi']:
        args.patch_size = args.patch_size[0]
        data_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
        npz_paths = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        if args.n_patient >= 0: 
            npz_paths = [npz_paths[args.n_patient]]
        for npz_path in npz_paths:
            npz_path = os.path.join(data_dir, npz_path)

            labels = np.unique(np.load(npz_path)["labels"])
            if args.N_UP in [-100, -50]: 
                sparse = False
                if args.sparse_method != 'None': 
                    sparse = True
                startttt = time.time()
                label_propagation(args, npz_path, sparse)
                print(f'TIME for all iterative message passing inferences: {time.time()-startttt} seconds')
            else:
                startttt = time.time()
                inference_iteration(args, npz_path)
                print(f'TIME for all iterative message passing inferences: {time.time()-startttt} seconds')

            args.annotations = []
            args.label_annotation = []
            args.weight_initial = None
            args.annotated_row_positions = []

    else:
        args.patch_size = args.patch_size[0]
        data_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
        npz_path = os.path.join(data_dir, f'{args.dataset}_{args.model}.npz')
        if args.pair_pot[0] == 'model_features': 
            npz_path = os.path.join(data_dir, f'{args.dataset}_{args.model}_similar.npz')
        if args.N_UP in [-100, -50]: 
            # -100 for LP hard
            # -50 for LP soft
            sparse = False
            if args.sparse_method != 'None': 
                sparse = True
            startttt = time.time()
            label_propagation(args, npz_path, sparse)
            print(f'TIME for label propagation: {time.time()-startttt} seconds')
        else:
            startttt = time.time()
            inference_iteration(args, npz_path)
            print(f'TIME for all iterative message passing inferences: {time.time()-startttt} seconds')
