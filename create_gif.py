import argparse
import json
import gif 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os 

#from plot_results import reconstruct
from plot_results_final import mask_to_centers, reconstruct


# Custom palette
my_colors = [
    "#f1c4d5",  
    "#4ec15b", 
    "#e77bf7", 
    "#480848",  
    "#e0e0e0",  
]
cmap3 = mcolors.ListedColormap(my_colors)


def visualize(results, npz_path):
    # Define the directory containing the features 
    root_dir = '/auto/globalscratch/users/t/g/tgodelai/test'
    data_dir = os.path.join(root_dir, 'data_processed', args.dataset, args.model)
    npz_name = f'{args.dataset}_{args.model}_{args.n_patient}' 
    wsi_path = f'/auto/globalscratch/users/t/g/tgodelai/thunder/datasets/bach/ICIAR2018_BACH_Challenge/WSI/thumbnails/A0{str(args.n_patient)}_thumb.png'

    npz_path = os.path.join(data_dir, npz_name+'.npz')

    # Read npz
    data = np.load(npz_path, allow_pickle=True)

    positions = data['positions']

    labels = data['labels']
    #probabilities = np.argmax(data['probabilities'], axis=1)

    #final_label = data["final_label"]
    #final_label = [f for i,f in enumerate(final_label) if i in [0, 1, 2, 3]]# [0, 7]]


    _, label_map = reconstruct(positions, results)

    return label_map


def create_gif(args): 
    root_dir = '/auto/globalscratch/users/t/g/tgodelai/test'
    data_dir = os.path.join(root_dir, 'data_processed', args.dataset, args.model)
    npz_name = f'{args.dataset}_{args.model}_{args.n_patient}' 
    npz_path = os.path.join(data_dir, npz_name+'.npz')

    weights = args.weight
    weight = [str(w) for w in weights]
    weight = ','.join(weight)   
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
    #json_path = os.path.join(root_dir, args.dataset, args.model, json_name)
    json_path = '/auto/globalscratch/users/t/g/tgodelai/test/results/bach_wsi/conchanduni2/gif_50_-1_5_10_0.1,0.01_1.0_minusmodelfeatures,modelfeaturesann_indicator,potts_0.1_cossim_[0, 16]_5_0_2_circle_False_bach_wsi_conchanduni2_5.json'

    with open(json_path, "r") as f:
        results = json.load(f)
    
    final_labels = results["final_label"]

    @gif.frame
    def plotgif(i):
        plt.imshow(visualize(final_labels[i], npz_path), cmap=cmap3, interpolation='nearest')
        plt.axis("off")

        #return plt.gcf()

    propagation_steps = []
    for i in [4]:
        p = np.arange(i*50,i*50+50)
        propagation_steps.extend(p)
    #np.arange(25) + np.arange(50, 50+25) + + np.arange(100, 100+25)
    print("propagation_steps", propagation_steps)
    frames = [plotgif(i) for i in propagation_steps]
    print("len(frames)", len(frames))

    gif.save(frames, 'example.gif', duration=150)


def parser():
    pair_pot_choices = ['image_features', 'model_features', 
                        'image_and_position', 'image', 
                        'hsv_image_and_position', 'binary_mask', 
                        'hsl_image_and_position', 
                        'cielab_image_and_position', 'edges',
                        'model_features_and_position', 
                        'cos_sim', 'patch', 'prob',
                        'minus_model_features', 'model_features_ann']
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

create_gif(args)