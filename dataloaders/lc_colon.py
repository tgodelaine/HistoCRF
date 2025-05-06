import os
import pandas as pd
from PIL import Image
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

labels_dict = {"colon_n": 0, "colon_aca": 1}

class LCcolon(Dataset):
    
    dataset_dir = "LC25000"
    
    def __init__(self, root, transform=None):

        self.dataset_dir = os.path.join(root, 'data', self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'colon_image_sets')
        
        self.images = self.get_images()

        self.classnames = ['Benign colonic tissue', 'Colon adenocarcinomas']

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
        for c in ['colon_n', 'colon_aca']:
            class_dir = os.path.join(self.image_dir, c)
            images_c = os.listdir(class_dir)
            images_c = [os.path.join(class_dir, i) for i in images_c if i.endswith('.jpeg')]
            images.extend(images_c)
        return images