import numpy as np
import os
from PIL import Image
import re
from torchvision.io import read_image
from torch.utils.data import Dataset 

templates = [
            "a histopathological image showing {}.",
            "{}.",
            "a photomicrograph showing {}.",
            "a photomicrograph of {}.",
            "an image of {}.",
            "an image showing {}.",
            "an example of {}.",
            "{} is shown.",
            "this is {}.",
            "there is {}.",
            "a histopathological image of {}.",
            "a histopathological photograph of {}.",
            "a histopathological photograph showing {}.",
            "shows {}.",
            "presence of {}.",
            "{} is present.",
            "an H&E stained image of {}.",
            "an H&E stained image showing {}.",
            "an H&E image showing {}.",
            "an H&E image of {}.",
            "{}, H&E stain.",
            "{}, H&E."
]

labels_dict = {"Healthy": 0, "Benign": 1, 
                "Carcinoma_in": 2, "Carcinoma_invasive": 3,
                "Background": 4, "Invasive_carcinoma": 3, "Invasive": 3, "In": 2}


def extract_coords(name):
    # Assumes the format ends with `_x_y_label_l_000`
    match = re.search(r'tile_\w+_\d+_(\d+)_(\d+)_label_\d+_\d+\.png', name)
    if match:
        x, y = map(int, match.groups())
        return (x, y)
    else:
        return (float('inf'), float('inf'))  # In case of invalid format

class BACH_WSI(Dataset):
    
    dataset_dir = os.path.join("thunder", "datasets", "bach")
    
    def __init__(self, root, transform=None, n_patient=2):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'ICIAR2018_BACH_Challenge', 'patches_512')

        data_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        data_files = [f for f in data_files if '_A0'+str(n_patient) in f]

        self.background = False # Remove background if False 

        self.classnames = ['Normal', 'Benign', 'In Situ carcinoma', 'Invasive carcinoma', 'An empty glass slide']

        self.template = templates
        self.transform = transform

        if not self.background:
            data_files_without_background = []
            for f in data_files:
                l = f.split('_')[5]
                if l == 'Carcinoma':
                    l = f.split('_')[5] + '_' + f.split('_')[6]
                label = labels_dict[l.split('.')[0]]
                if label != 4:
                    data_files_without_background.append(f)
            data_files = np.copy(data_files_without_background)
        self.data_files = sorted(data_files, key=extract_coords)
        if not self.background: 
            self.classnames = self.classnames[:-1]

    def __len__(self):
        return len(self.data_files)

    def  __getitem__(self, idx):
        image_name = self.data_files[idx]
        image_path =  os.path.join(self.image_dir, image_name)
        image = read_image(image_path)

        l = image_name.split('_')[5]
        if l == 'Carcinoma':
            l = image_name.split('_')[5] + '_' + image_name.split('_')[6]
        label = labels_dict[l.split('.')[0]] 

        x, y = image_name.split('_')[2:4]
        position = [int(x),int(y)]

        if self.transform:
            image = Image.open(image_path)
            image = self.transform(image)

        return image, label, position