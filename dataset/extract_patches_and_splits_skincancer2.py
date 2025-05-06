import argparse 
import cv2
import glob 
import json 
import numpy as np
import os
from PIL import Image
import random
import re
import shutil
import tifffile as tiff
from tqdm import tqdm


# Define the RGB to label mapping
COLOR_TO_LABEL = {
    (0, 0, 0): 0,
    (108, 0, 115): 1,
    (145, 1, 122): 2,
    (216, 47, 148): 3,
    (254, 246, 242): 4,
    (181, 9, 130): 5,
    (236, 85, 157): 6,
    (73, 0, 106): 7,
    (248, 123, 168): 8,
    (127, 255, 255): 9,
    (127, 255, 142): 10,
    (255, 127, 127): 11
}

def convert_rgb_mask_to_label(mask_path):
    """
    Convert an RGB segmentation mask to a grayscale mask with integer labels.
    
    Args:
        mask_path (str): Path to the RGB segmentation mask (.png)
    
    Returns:
        numpy.ndarray: Grayscale mask with integer labels from 0 to 11
    """
    # Load the RGB mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)  # OpenCV loads as BGR
    
    # Convert BGR to RGB (since OpenCV loads in BGR format)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Create an empty label mask
    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    # Assign labels based on the RGB values
    for rgb, label in COLOR_TO_LABEL.items():
        # Find pixels matching this color
        matches = (mask[:, :, 0] == rgb[0]) & (mask[:, :, 1] == rgb[1]) & (mask[:, :, 2] == rgb[2])
        label_mask[matches] = label

    return label_mask

# Function to extract tiles
def extract_tiles(args, image_path, mask_path, tile_dir, tile_size=256, is_train=True):
    # Read the whole-slide image (WSI)
    wsi = tiff.imread(image_path)  # Shape: (H, W, C) for RGB images
    image_name = image_path.split(os.sep)[-1].split(args.file_ending)[0]
    mask = convert_rgb_mask_to_label(mask_path) 

    height, width, _ = wsi.shape  # Get WSI dimensions

    # Set step size: 50% overlap for training, no overlap for testing
    step_size = tile_size // 2 if is_train else tile_size

    # Iterate over the image in tile_size steps
    for x in tqdm(range(0, width - tile_size + 1, step_size)):
        for y in range(0, height - tile_size + 1, step_size):
            # Extract tile from WSI
            tile = wsi[y:y+tile_size, x:x+tile_size]

            # Extract corresponding mask region
            mask_tile = mask[y:y+tile_size, x:x+tile_size]

            # Ensure tile is the correct size (skip if out of bounds)
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                
                # Compute percentage of background pixels (assuming background is labeled as 0)
                background_ratio = np.sum(mask_tile == 0) / mask_tile.size
                
                # Skip tiles with more than 50% background
                if background_ratio <= BACKGROUND_THRESHOLD:
                    # Determine label based on mask majority class
                    unique_labels, counts = np.unique(mask_tile, return_counts=True)
                    label = unique_labels[np.argmax(counts)]  # Majority label
                    
                    # Convert to PIL Image and save
                    tile_image = Image.fromarray(tile)
                    tile_filename = f"{tile_dir}/tile_{image_name}_{x}_{y}_label_{label}.png"
                    tile_image.save(tile_filename)
                    print("save file to {tile_filename}")

                    # Convert to PIL Image and save
                    mask_tile_image = Image.fromarray(mask_tile)
                    mask_tile_filename = f"{mask_tile_dir}/tile_{image_name}_{x}_{y}_label_{label}.png"
                    mask_tile_image.save(mask_tile_filename)
                    print("save mask file to {mask_tile_filename}")

'''
def make_splits(args, train_image_dest, train_label_dest, test_image_dest, test_label_dest, train_file, test_file, image_source_dir, label_source_dir):
    # Process all images
    for filename in os.listdir(image_source_dir):
        if filename.endswith(args.file_ending):
            image_path = os.path.join(image_source_dir, filename)
            mask_path = os.path.join(label_source_dir, filename.replace(args.file_ending, ".png"))
            is_train = filename in train_file
            extract_tiles(args, image_path, mask_path, train_image_dest if is_train else test_image_dest, args.tile_size, is_train)
'''

def make_splits(args, train_image_dest, train_label_dest, test_image_dest, test_label_dest, train_file, test_file, image_source_dir, label_source_dir):
    # Dictionary to store tiles by original image name
    train_tiles_by_label = {}
    test_tiles_by_label = {}

    # Regex pattern to extract original image name and label
    pattern = re.compile(r"tile_(.+?)_\d+_\d+_label_(\d+)\.png")

    # Organize tiles by original image name and class label
    for file_name in os.listdir(image_source_dir):
        match = pattern.match(file_name)
        if match:
            original_name, class_label = match.groups()
            if original_name in train_file:
                train_tiles_by_label.setdefault(class_label, []).append(file_name)
            elif original_name in test_file:
                test_tiles_by_label.setdefault(class_label, []).append(file_name)

    # Shuffle the tiles for randomness
    for label in train_tiles_by_label:
        random.shuffle(train_tiles_by_label[label])
    for label in test_tiles_by_label:
        random.shuffle(test_tiles_by_label[label])

    # Select up to n_train_tile train tiles per class
    train_selected = [tile for _, tiles in train_tiles_by_label.items() for tile in tiles] #SELECT ALL IMAGES IN THE TRAINING SET 
    ''' DECOMMENT TO LIMIT THE NUMBER OF IMAGES FOR EACH CLASS IN THE TRAINING SET
    train_selected = []
    for label, tiles in train_tiles_by_label.items():
        train_selected.extend(tiles[:min(args.n_train_tile, len(tiles))])'
    '''
    test_selected = [tile for _, tiles in test_tiles_by_label.items() for tile in tiles]
    print("test_selected", test_selected)

    def move_files(file_list, source_dir, dest_folder, label=False):
        for file_name in file_list:
            print(f"Process {file_name}")
            src = os.path.join(source_dir, file_name)
            if label:
                dest = os.path.join(dest_folder, file_name)
            else:
                dest = os.path.join(dest_folder, file_name.replace('.png', '_0000.png'))
            if os.path.exists(src): 
                shutil.copy2(src, dest)
            else:
                print(f'{src} existance ? {os.path.exists(src)}')


    # Move images and labels for training and testing
    if not args.test_only: 
        move_files(train_selected, image_source_dir, train_image_dest)
    move_files(test_selected, image_source_dir, test_image_dest)

    if not args.test_only: 
        move_files(train_selected, label_source_dir, train_label_dest, label=True)
    move_files(test_selected, label_source_dir, test_label_dest, label=True)

    print(f"Moved {len(train_selected)} training and {len(test_selected)} test tiles.")
    print("File transfer complete!")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/auto/globalscratch/users/t/g/tgodelai/miccai25/data/')
    parser.add_argument('--dataset', type=str, default='skincancer2', choices=['skincancer2'])
    parser.add_argument('--tile_size', type=int, default=254)
    parser.add_argument('-thresh', '--background_threshold', type=float, default=0.9, help='threshold for background filtering')
    parser.add_argument('--file_ending', type=str, default='.tif')
    parser.add_argument('--n_train_tile', type=int, default=100, help='Number of train tiles per class')
    parser.add_argument('--test_only', type=bool, default=False, help='If true, process only test images')
    args = parser.parse_args()

    return args


def create_json(args, train_image_dest, dataset_dest):
    # Count the number of training images
    n_train_file = len([f for f in os.listdir(train_image_dest) if f.endswith('.png')])

    if args.dataset == 'skincancer2':
        json_data = {
            "name": "skincancer2",
            "description": "Segmentation dataset for skin cancer WSIs.",
            "channel_names": {
                "0": "rgb_to_0_1",
                "1": "rgb_to_0_1",
                "2": "rgb_to_0_1"
            },
            "labels": {
                "background": 0,
                "class 1": 1,
                "class 2": 2,
                "class 3": 3,
                "class 4": 4,
                "class 5": 5,
                "class 6": 6,
                "class 7": 7,
                "class 8": 8,
                "class 9": 9,
                "class 10": 10,
                "class 11": 11
            },
            "numTraining": n_train_file,
            "file_ending": ".png"
        }

        # Save JSON to file
        json_path = os.path.join(dataset_dest, "dataset.json")
        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"JSON file saved at: {json_path}")


if __name__ == '__main__':
    args = parser()

    # Define paths
    data_dir = args.data_dir
    if args.dataset == 'skincancer2':
        data_dir = os.path.join(args.data_dir, 'skincancer2/data/1x') 
        name_dataset = f'Dataset{str(args.tile_size)}_{args.dataset}'
    image_dir, mask_dir = os.path.join(data_dir, 'Images'), os.path.join(data_dir, f'Masks')
    tile_dir, mask_tile_dir  = os.path.join(data_dir, name_dataset, 'Tiles'), os.path.join(data_dir, name_dataset, f'Tiles_mask')
    train_txtfile = os.path.join(data_dir, 'train.txt')
    test_txtfile = os.path.join(data_dir, 'test.txt')
    with open(train_txtfile, 'r') as f: 
        train_file = f.readlines()
        train_file = [f.replace('\n', '') for f in train_file]
    with open(test_txtfile, 'r') as f: 
        test_file = f.readlines()
        test_file = [f.replace('\n', '') for f in test_file]
    print("test_file", test_file)

    # Ensure tile directory exists and if exist, remove existing files 
    os.makedirs(tile_dir, exist_ok=True)
    os.makedirs(mask_tile_dir, exist_ok=True)

    '''
    for file in glob.glob(os.path.join(tile_dir, args.file_ending)):
        os.remove(file)
    for file in glob.glob(os.path.join(mask_tile_dir, '*.png')):
        os.remove(file)
    ''' 
    # Tile size
    TILE_SIZE = args.tile_size
    BACKGROUND_THRESHOLD = args.background_threshold

    # Process all images
    for filename in os.listdir(image_dir):
        if filename.endswith(args.file_ending):
            if args.test_only: 
                if (filename.replace(args.file_ending, '')) in test_file: 
                    image_path = os.path.join(image_dir, filename)
                    mask_path = os.path.join(mask_dir, filename.replace(args.file_ending, ".png"))  # Ensure mask name matches
                    print(f"Test only! Extract {image_path} and {mask_path} files")
                    extract_tiles(args, image_path, mask_path, tile_dir, args.tile_size, False)
            else: 
                image_path = os.path.join(image_dir, filename)
                mask_path = os.path.join(mask_dir, filename.replace(args.file_ending, ".png")) # Ensure mask name matches
                is_train = filename in train_file
                print(f"Not test only! Extract {image_path} and {mask_path} files")
                extract_tiles(args, image_path, mask_path, tile_dir, args.tile_size, is_train)

    # Define destination directories
    train_image_dest = os.path.join(data_dir, name_dataset, f'imagesTr')
    train_label_dest = os.path.join(data_dir, name_dataset, f'labelsTr')
    test_image_dest = os.path.join(data_dir, name_dataset, f'imagesTe')
    test_label_dest  = os.path.join(data_dir, name_dataset, f'labelsTe')
    print("test_label_dest", test_label_dest)

    # Ensure destination directories exist
    for dir_path in [train_image_dest, train_label_dest, test_image_dest, test_label_dest]:
        os.makedirs(dir_path, exist_ok=True)

    '''
    if not args.test_only: 
        for file in glob.glob(os.path.join(train_image_dest,  '*.png')):
            os.remove(file)
        for file in glob.glob(os.path.join(train_label_dest, '*.png')):
            os.remove(file)
    for file in glob.glob(os.path.join(test_image_dest,  '*.png')):
        os.remove(file)
    for file in glob.glob(os.path.join(test_label_dest, '*.png')):
        os.remove(file)
    '''
    print("---- Make split -----")
    make_splits(args, train_image_dest, train_label_dest, test_image_dest, test_label_dest, train_file, test_file, tile_dir, mask_tile_dir)

    print("---- Make json -----")
    dataset_dest = os.path.join(data_dir, name_dataset)
    create_json(args, train_image_dest, dataset_dest)
    
    source_folder = dataset_dest
    destination_folder = os.path.join(args.data_dir, 'nnUNet_raw', name_dataset)
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder) # Remove destination folder if it exists
    shutil.copytree(source_folder, destination_folder) # Copy the folder