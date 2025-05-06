import os
import random
import pandas as pd
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset 
import torch 
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

labels_dict = {"lung_aca": 0, "lung_n": 1, 
                "lung_scc": 2}

class LClung(Dataset):
    
    dataset_dir = "LC25000"
    
    def __init__(self, root, transform=None):

        self.dataset_dir = os.path.join(root, 'data', self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'lung_image_sets')
        
        self.images = self.get_images()

        self.classnames = ['Lung adenocarcinoma', 'Benign lung’', 'Lung squamous cell carcinoma']

        self.template = templates
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def  __getitem__(self, idx):
        image_path = self.images[idx]
        image = read_image(image_path)
        label = labels_dict[image_path.split(os.sep)[-2]]
        classname = self.classnames[label]

        if self.transform:
            image = Image.open(image_path)
            image = self.transform(image)

        return image, label
    
    def get_images(self):
        images = []
        for c in ['lung_aca', 'lung_n', 'lung_scc']:
            class_dir = os.path.join(self.image_dir, c)
            images_c = os.listdir(class_dir)
            images_c = [os.path.join(class_dir, i) for i in images_c if i.endswith('.jpeg')]
            images.extend(images_c)
        return images