import os
import pandas as pd
from PIL import Image
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
            #"a histopathological image showing {}.",
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


labels_dict = {"TUM": 8, "STR": 7, "NORM": 6, "MUS": 5, "MUC": 4, "LYM": 3, "DEB": 2, "BACK": 1, "ADI": 0}


class NCT(Dataset):
    
    dataset_dir = "NCT-CRC-HE-100K"

    def __init__(self, root, transform=None):

        self.dataset_dir = os.path.join(root, 'data', self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "NCT-CRC-HE-100K")
        self.image_dir_test = os.path.join(self.dataset_dir, "CRC-VAL-HE-7K")

        self.images = self.get_images(self.image_dir)    
        self.images_test = self.get_images(self.image_dir_test)    

        self.classnames = ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle",
           "Normal ; mucosa", "Cancer-associated stroma", 
           "Colorectal adenocarcinoma epithelium"]

        self.template = templates
        self.transform = transform

    def __len__(self, test=True):
        if test:
            return len(self.images_test)
        else:
            return len(self.images)

    def  __getitem__(self, idx, test=True):
        if test: 
            img_path = self.images_test[idx]
        else: 
            img_path = self.images[idx]
        image = Image.open(img_path)
        label = labels_dict[img_path.split(os.sep)[-2]]
        classname = self.classnames[label]

        if self.transform:
            image = Image.open(img_path)
            image = self.transform(image)

        return image, label
    
    def get_images(self, images_dir):
        images = []
        for c in ['TUM', 'STR', 'NORM', 'MUS', 'MUC', 'LYM', 'DEB', 'BACK', 'ADI']:
            class_dir = os.path.join(images_dir, c)
            images_c = os.listdir(class_dir)
            images_c = [os.path.join(class_dir, i) for i in images_c if i.endswith('.tif')]
            images.extend(images_c)
        return images