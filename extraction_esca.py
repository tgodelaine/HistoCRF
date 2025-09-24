import numpy as np 
import os
import torch


def extraction_esca(args):
    npz_name = f'{args.dataset}_{args.model}.npz'
    npz_path = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model, npz_name)
    indices_high_affinities = np.load(npz_path)['cossim']

    indices_high_affinities = torch.tensor(indices_high_affinities)
    row_lengths = indices_high_affinities.sum(dim=1)        # number of True per row
    max_len = row_lengths.max().item()
    n = indices_high_affinities.shape[0]
    # Build padded tensor by repeating values
    padded = torch.empty((n, max_len), dtype=torch.int16, device='cuda')
    for i in range(n):
        valid = torch.nonzero(indices_high_affinities[i]).squeeze(1)
        repeat_factor = (max_len + valid.numel() - 1) // valid.numel()  # ceil division
        repeated = valid.repeat(repeat_factor)[:max_len]                # repeat & trim
        padded[i] = repeated

    npz_name_save = f'{args.dataset}_{args.model}_cossim.npz'
    npz_path_save = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model, npz_name_save)
    np.savez_compressed(
        npz_path_save,
        cossim=padded.detach().cpu().numpy()
    )
    print(f"Save to {npz_name_save}")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='sicap_mil', choices=['sicap_mil', 'monuseg', 'skincancer', 'panuke', 'skincancer2', 'lc_lung', 'lc_colon', 'nct', 'tcga-ut', 'bracs', 'bach', 'esca', 'tcga_uniform', 'bach_wsi'])
    parser.add_argument('--model', type=str, default='clip', choices=['clip', 'quilt', 'plip', 'conch', 'nnunet', 'nnunet_patches', 'nnunet_bigpatches', 'plipanduni2', 'conchanduni2', 'conchandgigapath', 'conchandvirshow2', 'conchandoptimus1'])
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    parser.add_argument('--file_ending', type=str, default='.tif')
    parser.add_argument('--patch_size', type=int, default=254)
    parser.add_argument('--n_patient', type=int, default=0)
    parser.add_argument('--dissimilar', type=bool, default=True)
    args = parser.parse_args()

    return args


