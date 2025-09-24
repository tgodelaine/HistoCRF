import os
from PIL import Image
import re
from torchvision.io import read_image
from torch.utils.data import Dataset 
import numpy as np 


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


def extract_coords(name):
    # Assumes the format ends with `_x_y_label_l_000`
    match = re.search(r'tile_\w+_\d+_(\d+)_(\d+)_label_\d+_\d+\.png', name)
    if match:
        x, y = map(int, match.groups())
        return (x, y)
    else:
        return (float('inf'), float('inf'))  # In case of invalid format

class SkinCancer(Dataset): 
    
    def __init__(self, root, transform=None, n_patient=0):

        self.image_dir = os.path.join(root, 'data', 'nnUNet_raw', 'Dataset254_skincancer2', 'imagesTe')
        txt_path = '/auto/globalscratch/users/t/g/tgodelai/miccai25/data/skincancer2/data/1x/test.txt'
        with open(txt_path, 'r') as f:
            test_patients = f.readlines()
        patient = test_patients[n_patient].split('\n')[0]
        print("patient", patient)

        data_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        data_files = [f for f in data_files if patient + '_' in f]
        self.background = False # Remove background if False 
        if not self.background:
            data_files_without_background = []
            for f in data_files:
                label = int(f.split('_')[6])
                if label != 0:
                    data_files_without_background.append(f)
            data_files = np.copy(data_files_without_background)
        print("len data_files", len(data_files))

        self.data_files = sorted(data_files, key=extract_coords)

        self.classnames = [
                "An empty glass slide",
                "Glands",
                "Inflammation",
                "Hair Follicle",
                "Hypodermis",
                "Reticular Dermis",
                "Papillary Dermis",
                "Epidermis", 
                "Keratin",
                "Basal Cell Carcinoma",
                "Squamous Cell Carcinom",
                "Intra Epidermal Carinoma"
             ]
        if not self.background: 
            self.classnames = self.classnames[1:]
        self.template = templates
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def  __getitem__(self, idx):
        #img_path = os.path.join(self.image_dir, 'data', '1x', 'Tiles', self.data_test.at[idx, "Tiles_path"])
        image_name = self.data_files[idx]
        image_path =  os.path.join(self.image_dir, image_name)
        image = read_image(image_path)
        #w, h, c = image.shape
        label = int(image_name.split('_')[6])   #int(self.data_test.at[idx, "Label"])
        if not self.background: 
            label = label - 1
        x, y = image_name.split('_')[3:5]
        position = [int(x),int(y)]
        #classname = self.classnames[self.data_test.at[idx, "labels"]]

        if self.transform:
            image = Image.open(image_path)
            image = self.transform(image)

        return image, label, position