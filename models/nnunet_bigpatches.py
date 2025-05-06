import os 
import numpy as np

dataset_name_conversion = {
    'skincancer2': 'Dataset254_skincancer2',
}


def extract_patches(npz_file, patch_size):
    """
    Extracts non-overlapping patches from a segmentation map stored in an npz file.

    Parameters:
        npz_file (str): Path to the npz file containing the segmentation map.
        patch_size (int): Size of the square patches to extract.

    Returns:
        patches (list): A list of NumPy arrays representing the extracted patches.
        names (list): A list of patch names formatted as 'patch_x_y'.
    """
    
    # Load the segmentation map from the npz file
    data = np.load(npz_file)
    segmentation_map = data['segmentation_map']  # Ensure this is the correct key

    patches, names = [], []
    dim, height, width = segmentation_map.shape  # Get the dimensions of the segmentation map

    # Iterate over the image in steps of patch_size
    for y in range(0, height, patch_size[1]):
        for x in range(0, width, patch_size[0]):
            # Extract patch
            patch = segmentation_map[:, y:y+patch_size[1], x:x+patch_size[0]]

            # Ensure patch is the correct size (in case the image dimensions are not exact multiples of patch_size)
            if patch.shape == (dim, patch_size[1], patch_size[0]):
                patches.append(np.transpose(patch, (1, 2, 0)))
                names.append(f"patch_{x}_{y}")

    return patches, names


def features_extraction_nnunet(args, test_file):
    npz_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model, str(args.patch_size)) #?????? args.patch_size????
    if not os.path.isdir(npz_dir):
        os.makedirs(npz_dir)
    dataset_dir = os.path.join(args.data_dir, 'nnUNet_raw', dataset_name_conversion[args.dataset])
    images_dir = os.path.join(dataset_dir, 'imagesTe')
    labels_dir = os.path.join(dataset_dir, 'labelsTe')
    predictions_dir = os.path.join(dataset_dir, 'predictionsTe')

    for f in test_file:
        if os.path.isfile(os.path.join(images_dir, f'reconstructed_{f}.npz')):
            images, names = extract_patches(os.path.join(images_dir, f'reconstructed_{f}.npz'), args.patch_size)
            labels, _ = extract_patches(os.path.join(labels_dir, f'reconstructed_{f}.npz'), args.patch_size)
            predictions, _ = extract_patches(os.path.join(predictions_dir, f'reconstructed_{f}.npz'), args.patch_size)

            for img, lab, pred, name in zip(images, labels, predictions, names):
                n_class = pred.shape[-1]
                width, height = lab.shape[:2]
                pred = np.transpose(pred, (2, 0, 1))
                print("pred", pred[:,0,0], pred[:,50,50], pred[::,-1,-1])
                # Save the numpy arrays in an .npz file
                npz_name = f'{args.dataset}_nnunet_patches_{f}_{name}'
                npz_path = os.path.join(npz_dir, npz_name)
                np.savez_compressed(
                    npz_path, 
                    width=width, 
                    height=height, 
                    n_labels=n_class, 
                    images=img, 
                    labels=lab, 
                    probabilities=pred, 
                    )

                print(f"Saved features for {img} to {npz_path}")

    return "Feature extraction and saving completed."

# Test usage 
import argparse 

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='sicap_mil', choices=['sicap_mil', 'monuseg', 'skincancer', 'panuke', 'skincancer2'])
    parser.add_argument('--model', type=str, default='clip', choices=['clip', 'quilt', 'plip', 'conch', 'nnunet', 'nnunet_patches'])
    parser.add_argument('--file_ending', type=str, default='.tif')
    parser.add_argument('--patch_size', type=int, nargs='+', default=224)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parser()

    args.data_dir = '/auto/globalscratch/users/t/g/tgodelai/miccai25/data/'
    args.root_dir = '/CECI/home/users/t/g/tgodelai/miccai25'
    args.dataset = 'skincancer2'
    args.model = 'nnunet_bigpatches'
    args.file_ending = '.png'

    if args.dataset == 'skincancer2':
        test_txtfile = os.path.join(args.data_dir, 'skincancer2', 'data', '1x', 'test.txt')
        with open(test_txtfile, 'r') as f: 
            test_file = f.readlines()
            print("f1", f)
            test_file = [f.replace('\n', '').split(".")[0] for f in test_file]
    else:
        raise RuntimeError(f"Not implemented yet for dataset {args.dataset}")

    features_extraction_nnunet(args, test_file)