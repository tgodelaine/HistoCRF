import argparse 
import matplotlib.pyplot as plt 
import numpy as np
import os
from PIL import Image
import re
from scipy.special import softmax
from glob import glob

SANITY_CHECK = True
VISUALIZE = False

def reconstruct_segmentation_map(tile_files, output_path, tile_size=224, n_labels=12, file_ending= "*.npz"):
    """
    Reconstructs the segmentation mask of a whole slide image from tiled npz masks.
    
    Parameters:
        tile_dir (str): Path to the directory containing the npz tile masks.
        output_path (str): Path to save the reconstructed whole slide npz file.
        tile_size (int): Size of each tile (default: 224).
    """
    
    # Step 2: Extract coordinates from filenames and find image dimensions
    tile_data = []
    max_x, max_y = 0, 0

    for file in tile_files:
        # Extract x, y coordinates from filename using regex
        match = re.search(r"_(\d+)_(\d+)_label", os.path.basename(file))
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            if file_ending == "*.npz":
                logits = np.load(file, allow_pickle=True)["probabilities"]  # Assuming segmentation mask 
                n_class = logits.shape[0]
                width, height = logits.shape[2], logits.shape[3]
                data = softmax(logits, axis=0).reshape((n_class, width, height)) # Apply softmax to compute probabilities

            elif file_ending == '*.png':
                if n_labels == 3:
                    data = np.transpose(np.array(Image.open(file).convert('RGB')), (2, 0, 1))
                else:
                    data = np.array(Image.open(file))

            tile_data.append((x, y, data))
            
            # Determine the required canvas size
            max_x = max(max_x, x + tile_size)
            max_y = max(max_y, y + tile_size)

    # Step 3: Initialize an empty segmentation map
    segmentation_map = (1/12)*np.ones((n_labels, max_y, max_x), dtype=np.float16)

    # Step 4: Place each tile into the correct position
    for x, y, mask in tile_data:
        segmentation_map[:, y:y+tile_size, x:x+tile_size] = mask.reshape((n_labels, tile_size, tile_size))

    print("segmentation_map", segmentation_map[segmentation_map != 0])
    np.savez_compressed(output_path, segmentation_map=segmentation_map)

    plt.figure()
    if n_labels not in [1, 3]:
        a = np.argmax(segmentation_map.transpose(1, 2, 0), axis=2).astype(np.int64)
        a = a.reshape(a.shape[0], a.shape[1], 1)
        plt.imshow(a)
    else: 
        plt.imshow(segmentation_map.transpose(1, 2, 0).astype(np.int64))

    plt.savefig(output_path.split(".npz")[0]+'.png')

    # Step 5: Save the reconstructed segmentation map
    #print("segmentation_map", segmentation_map)
    #np.savez_compressed(output_path, segmentation_map=segmentation_map)
    print(f"Reconstructed segmentation map saved to {output_path}")
    return segmentation_map


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/auto/globalscratch/users/t/g/tgodelai/miccai25/data/')
    parser.add_argument('--dataset', type=str, default='skincancer2', choices=['skincancer2'])
    parser.add_argument('--tile_size', type=int, default=224)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser()

    # Define paths
    data_dir = args.data_dir 
    if args.dataset == 'skincancer2':
        name_dataset = f'Dataset{str(args.tile_size)}_{args.dataset}'
        n_class = 12
        test_txtfile = os.path.join(data_dir, 'skincancer2', 'data', '1x', 'test.txt')
    with open(test_txtfile, 'r') as f: 
        test_file = f.readlines()
        test_file = [f.replace('\n', '') for f in test_file]

    test_image_dest = os.path.join(data_dir, 'nnUNet_raw', name_dataset, f'imagesTe')
    test_label_dest = os.path.join(data_dir, 'nnUNet_raw', name_dataset, f'labelsTe')
    test_pred_test = os.path.join(data_dir, 'nnUNet_raw', name_dataset, f'predictionsTe')

    for f in test_file:
        f = f.split('.')[0]
        print("f", f)

         # Step 1: Find all files in the directory
        test_image_files = glob(os.path.join(test_image_dest, "*.png"))
        test_image_files = [t for t in test_image_files if f+'_' in t and 'reconstructed' not in t]

        test_label_files = glob(os.path.join(test_label_dest, "*.png"))
        test_label_files = [t for t in test_label_files if f+'_' in t  and 'reconstructed' not in t]

        test_pred_files = glob(os.path.join(test_pred_test, "*.npz"))
        test_pred_files = [t for t in test_pred_files if f+'_' in t and 'reconstructed' not in t]
    
        # Directory
        test_image_out = os.path.join(test_image_dest, f'reconstructed_{f}.npz')
        test_label_out = os.path.join(test_label_dest, f'reconstructed_{f}.npz')
        test_pred_out = os.path.join(test_pred_test, f'reconstructed_{f}.npz')

        if len(test_image_files) > 0: 
            #print("Process test image")
            #reconstruct_segmentation_map(tile_files=test_image_files, output_path=test_image_out, tile_size=args.tile_size, n_labels=3, file_ending="*.png")
            #print("Process label")
            #reconstruct_segmentation_map(tile_files=test_label_files, output_path=test_label_out, tile_size=args.tile_size, n_labels=1, file_ending="*.png")
            print("Process pred")
            reconstruct_segmentation_map(tile_files=test_pred_files, output_path=test_pred_out, tile_size=args.tile_size, n_labels=n_class, file_ending="*.npz")

    '''
    tile_dirs = [test_image_dest, test_label_dest, test_pred_test]
    output_paths = [test_image_out, test_label_out, test_pred_out]
    file_endings = ["*.png", "*.png", "*.npz"]
    n_labels = [3, 1, n_class]

    for (tile_dir, output_path, file_ending, n_label) in zip(tile_dirs, output_paths, file_endings, n_labels):
        reconstruct_segmentation_map(tile_dir=tile_dir, output_path=output_path, tile_size=args.tile_size, n_labels=n_label, file_ending=file_ending)
    '''
'''
# Example usage
tile_directory = '/CECI/proj/medresyst/workshop_trail/data/nnUNet_raw/Dataset014_skincancer2/labelsTe'
output_file = '/CECI/proj/medresyst/workshop_trail/data/nnUNet_raw/Dataset014_skincancer2/test_reconstructed_label_file.npz'
segmentation_map = reconstruct_segmentation_map(tile_directory, output_file, 448, n_labels=1, file_ending= "*.png")
print("segmentation_map", segmentation_map.shape, segmentation_map[segmentation_map != 0])
'''